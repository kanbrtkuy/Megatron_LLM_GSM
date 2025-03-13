"""Sample Generate GPT"""
import os
import sys

import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),
                                             os.path.pardir)))
from megatron.training import print_rank_0
from megatron.core.models.gpt import GPTModel
from megatron.training.arguments import core_transformer_config_from_args
from megatron.training.yaml_arguments import core_transformer_config_from_yaml
from megatron.inference.text_generation_server import MegatronServer
from megatron.core.transformer.spec_utils import import_module
from megatron.core.models.gpt.gpt_layer_specs import (
    get_gpt_layer_local_spec,
    get_gpt_layer_with_transformer_engine_spec,
)
from megatron.training.utils import (
    get_batch_on_this_cp_rank,
    get_batch_on_this_tp_rank,
    get_blend_and_blend_per_split,
)
from megatron.core.datasets.gpt_dataset import GPTDatasetConfig

from contextlib import nullcontext
from typing import Union
import megatron
from argparse import Namespace
from torch.utils.data import DataLoader

from megatron.core.inference.model_inference_wrappers.inference_wrapper_config import InferenceWrapperConfig
from megatron.core.inference.engines.abstract_engine import AbstractEngine
from megatron.core.inference.engines.mcore_engine import MCoreEngine
from megatron.core.inference.model_inference_wrappers.gpt.gpt_inference_wrapper import GPTInferenceWrapper
from megatron.core.inference.text_generation_controllers.simple_text_generation_controller import SimpleTextGenerationController
from megatron.core.transformer.module import MegatronModule

from megatron.training import get_args
from megatron.training import get_tokenizer
from megatron.training.checkpointing import load_checkpoint
from megatron.core import mpu
from megatron.training.initialize import initialize_megatron
from megatron.training import get_model


from megatron.core.datasets.gpt_dataset import GPTDataset  #??
from megatron.core.datasets.indexed_dataset import IndexedDataset
from megatron.core.datasets.blended_megatron_dataset_builder import BlendedMegatronDatasetBuilder


test_num_samples =[48000, 0, 0]  ##can revise, sample = 48000/micro batchsize



def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def model_provider(pre_process=True, post_process=True) -> Union[GPTModel, 'megatron.legacy.model.GPTModel']:
    """Builds the model.

    Args:
        pre_process (bool, optional): Set to true if you need to compute embedings. Defaults to True.
        post_process (bool, optional): Set to true if you need to want to compute output logits/loss. Defaults to True.

    Returns:
        Union[GPTModel, megatron.legacy.model.GPTModel]: The returned model
    """
    args = get_args()
    use_te = args.transformer_impl == "transformer_engine" if hasattr(args, 'transformer_impl') else False

    print_rank_0('building GPT model ...')
    # Experimental loading arguments from yaml
    if hasattr(args, 'yaml_cfg') and args.yaml_cfg is not None:
        config = core_transformer_config_from_yaml(args, "language_model")
    else:
        config = core_transformer_config_from_args(args)

    if hasattr(args, 'use_legacy_models') and args.use_legacy_models:
        import megatron.legacy.model
        model = megatron.legacy.model.GPTModel(
            config,
            num_tokentypes=0,
            parallel_output=True,
            pre_process=pre_process,
            post_process=post_process,
        )
    else:  # using core models
        # 创建层重复模式
        repeat_pattern = None
        repeat_mode = os.environ.get('REPEAT_MODE', 'none')
        base_layers = args.num_layers
        
        if repeat_mode != 'none':
            if repeat_mode == 'every_layer':
                # 每层重复模式: 1234->11223344
                repeat_pattern = []
                for i in range(1, base_layers + 1):
                    repeat_pattern.extend([i, i])  # 每层重复两次
            
            elif repeat_mode == 'block':
                # 块重复模式: 1234->12341234
                block = list(range(1, base_layers + 1))
                repeat_pattern = block + block  # 整个块重复两次
            
            elif repeat_mode == 'alternate':
                # 交替重复模式
                repeat_pattern = [1]  # 首层
                for i in range(2, base_layers):
                    repeat_pattern.extend([i, i])  # 中间层重复
                repeat_pattern.append(base_layers)  # 末层
            
            elif repeat_mode == 'custom':
                # 自定义重复模式
                pattern_str = os.environ.get('REPEAT_PATTERN', '')
                if pattern_str:
                    try:
                        repeat_pattern = [int(x) for x in pattern_str.strip('[]').split(',')]
                    except ValueError:
                        print_rank_0(f"Warning: Could not parse custom repeat pattern '{pattern_str}'. Using no repetition.")
                        repeat_pattern = None
            
            print_rank_0(f"Using layer repetition mode: {repeat_mode}")
            print_rank_0(f"Repeat pattern: {repeat_pattern}")
            
            # 检查层索引是否在有效范围内
            if repeat_pattern:
                for idx in repeat_pattern:
                    if not (1 <= idx <= base_layers):
                        raise ValueError(f"Invalid layer index {idx} in repeat pattern. Must be between 1 and {base_layers}.")
        
        if hasattr(args, 'spec') and args.spec is not None:
            transformer_layer_spec = import_module(args.spec)
        else:
            if hasattr(args, 'num_experts') and args.num_experts:
                # 专家模型逻辑 - 如果不需要可以简化
                transformer_layer_spec = None  # 在实际代码中替换为正确的函数调用
            else:
                # 定义编码器层规格
                if use_te:
                    transformer_layer_spec = get_gpt_layer_with_transformer_engine_spec(
                        getattr(args, 'num_experts', 0), 
                        getattr(args, 'moe_grouped_gemm', False),
                        getattr(args, 'qk_layernorm', False), 
                        getattr(args, 'multi_latent_attention', False), 
                        getattr(args, 'moe_use_legacy_grouped_gemm', False))
                else:
                    transformer_layer_spec = get_gpt_layer_local_spec(
                        getattr(args, 'num_experts', 0), 
                        getattr(args, 'moe_grouped_gemm', False),
                        getattr(args, 'qk_layernorm', False), 
                        getattr(args, 'multi_latent_attention', False), 
                        getattr(args, 'moe_use_legacy_grouped_gemm', False))
        
        build_model_context = nullcontext
        build_model_context_args = {}
        
        # FP8参数收集逻辑 - 简化处理
        if hasattr(args, 'fp8_param_gather') and args.fp8_param_gather:
            print_rank_0("Warning: fp8_param_gather is set but not fully implemented in this script")
        
        with build_model_context(**build_model_context_args):
            model = GPTModel(
                config=config,
                transformer_layer_spec=transformer_layer_spec,
                vocab_size=args.padded_vocab_size,
                max_sequence_length=args.max_position_embeddings,
                pre_process=pre_process,
                post_process=post_process,
                fp16_lm_cross_entropy=getattr(args, 'fp16_lm_cross_entropy', False),
                parallel_output=True,
                share_embeddings_and_output_weights=not getattr(args, 'untie_embeddings_and_output_weights', False),
                position_embedding_type=getattr(args, 'position_embedding_type', 'learned_absolute'),
                rotary_percent=getattr(args, 'rotary_percent', 1.0),
                rotary_base=getattr(args, 'rotary_base', 10000.0),
                rope_scaling=getattr(args, 'use_rope_scaling', None),
                repeat_pattern=repeat_pattern
            )

    # 打印模型信息
    total_params = count_parameters(model)
    print_rank_0(f'Total trainable parameters: {total_params:,}')
    first_param = next(model.parameters())
    print_rank_0(f"Data type: {first_param.dtype}")
    
    return model


class GPTDatasetWrapper(torch.utils.data.Dataset):
    """包装 IndexedDataset 的类，适配数据格式"""
    def __init__(self, indexed_dataset, seq_length):
        self.indexed_dataset = indexed_dataset
        self.seq_length = seq_length
        
    def __len__(self):
        return len(self.indexed_dataset)
        
    def __getitem__(self, idx):
        tokens = self.indexed_dataset[idx]
        # 确保tokens是numpy数组
        if isinstance(tokens, tuple):
            tokens = tokens[0]  # 处理多模态情况
        
        # 确保tokens长度适合模型输入
        if len(tokens) > self.seq_length:
            tokens = tokens[:self.seq_length]
        
        return {'text': torch.from_numpy(tokens)}



def is_dataset_built_on_rank():
    return (
        mpu.is_pipeline_first_stage() or mpu.is_pipeline_last_stage()
    ) and mpu.get_tensor_model_parallel_rank() == 0





def main():
    # STEP 1 - 初始化模型并设置默认参数
    initialize_megatron(args_defaults={
        'no_load_rng': True,
        'no_load_optim': True,
        'micro_batch_size': 1
    })
    args = get_args()
    tokenizer = get_tokenizer()
    
    # STEP 2 - 加载模型
    model = get_model(model_provider, wrap_with_ddp=False)
    load_checkpoint(model, None, None)
    model = model[0]  # 获取实际模型
    model.eval()  # 设置为评估模式
    # import pdb;pdb.set_trace()
    
    # STEP 3 - 设置推理引擎
    inference_wrapper_config = InferenceWrapperConfig(
        hidden_size=args.hidden_size,
        inference_batch_times_seqlen_threshold=getattr(args, 'inference_batch_times_seqlen_threshold', 512),
        fp32_residual_connection=args.fp32_residual_connection,
        params_dtype=args.params_dtype,
        padded_vocab_size=args.padded_vocab_size,
        inference_max_seq_length=args.seq_length
    )
    
    inference_wrapped_model = GPTInferenceWrapper(model, inference_wrapper_config)
    text_generation_controller = SimpleTextGenerationController(
        inference_wrapped_model=inference_wrapped_model,
        tokenizer=tokenizer
    )
    
    # 加载训练数据集作为测试数据集
    
    
    # 创建索引数据集
    # import pdb;pdb.set_trace()
    # indexed_dataset = IndexedDataset(args.data_path[0], mmap=True)

    def core_gpt_dataset_config_from_args(args):
        tokenizer = get_tokenizer()

        # Sometimes --data-path is too long, instead we parse it from a file.
        blend: Optional[Tuple[List[str], Optional[List[float]]]]
        blend_per_split: Optional[List[Optional[Tuple[List[str], Optional[List[float]]]]]]
        blend, blend_per_split = get_blend_and_blend_per_split(args)

        return GPTDatasetConfig(
            random_seed=args.seed,
            sequence_length=args.seq_length,
            blend=blend,
            blend_per_split=blend_per_split,
            split=args.split,
            num_dataset_builder_threads=args.num_dataset_builder_threads,
            path_to_cache=args.data_cache_path,
            mmap_bin_files=args.mmap_bin_files,
            tokenizer=tokenizer,
            reset_position_ids=args.reset_position_ids,
            reset_attention_mask=args.reset_attention_mask,
            eod_mask_loss=args.eod_mask_loss,
            create_attention_mask=args.create_attention_mask_in_dataloader,
            s3_cache_path=args.s3_cache_path,
        )


    def train_valid_test_datasets_provider(train_val_test_num_samples):
        """Build the train test and validation datasets.

        Args:
            train_val_test_num_samples : A list containing the number of samples in train test and validation.
        """
        args = get_args()

        config = core_gpt_dataset_config_from_args(args)

        if args.mock_data:
            dataset_type = MockGPTDataset
        else:
            dataset_type = GPTDataset

        print_rank_0("> building train, validation, and test datasets for GPT ...")

        train_ds, valid_ds, test_ds = BlendedMegatronDatasetBuilder(
            dataset_type,
            train_val_test_num_samples,
            is_dataset_built_on_rank,
            config
        ).build()

        return train_ds, valid_ds, test_ds

    train_ds, _, _ = train_valid_test_datasets_provider(test_num_samples)

    
    
    # 创建包装数据集
    # train_dataset = GPTDatasetWrapper(indexed_dataset, args.seq_length)
    
    # 创建dataloader
    train_dataloader = DataLoader(
        train_ds, 
        batch_size=args.micro_batch_size,
        sampler=torch.utils.data.SequentialSampler(train_ds),
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )
    print_rank_0(f"Dataset loaded successfully, size: {len(train_dataloader)}")
    
    # 计算模型熵
    H_total = 0.0
    token_count = 0
    
    for batch_idx, batch in enumerate(tqdm(train_dataloader)):
        # 获取输入数据
        #######################
        tokens = batch['tokens'].cuda().int()
        attention_mask = torch.ones_like(tokens, dtype=torch.bool, device=tokens.device)
        
        # 准备位置ID
        batch_size, seq_length = tokens.size()
        position_ids = torch.arange(seq_length, dtype=torch.long, device=tokens.device)
        position_ids = position_ids.unsqueeze(0).expand_as(tokens)
        
        # 获取模型输出logits
        with torch.no_grad():
            outputs = model.forward(input_ids=tokens,position_ids=position_ids,attention_mask=attention_mask)
            
            # import pdb;pdb.set_trace()
        
        # 应用softmax获取概率分布
        probs = F.softmax(outputs, dim=-1)
        
        # 计算每个小批次的熵: Hi = sum(-Yi*log2(Yi))
        log_probs = torch.log2(probs + 1e-10)  # 添加小值避免log(0)
        entropy = -torch.sum(probs * log_probs, dim=-1)  # 按词汇表维度求和
        
        # 对序列和批次维度求和
        H_batch = entropy.sum().item()
        H_total += H_batch
        
        # 计算处理的token数量
        token_count += tokens.numel()
        
        # 每处理10个批次打印一次当前熵
        if batch_idx % 10 == 0 and torch.distributed.get_rank() == 0:
            print(f"Batch {batch_idx}, Current entropy: {H_total}, Tokens processed: {token_count}")
    
    # 收集所有进程的结果
    if torch.distributed.is_initialized():
        H_tensor = torch.tensor([H_total, token_count], dtype=torch.float64, device='cuda')
        torch.distributed.all_reduce(H_tensor)
        H_total, token_count = H_tensor.tolist()
    
    # 计算模型存储的信息
    total_information = 10_000_000  # 假设的总信息熵值
    model_stored_info = total_information - H_total
    remaining_info = H_total
    
    if torch.distributed.get_rank() == 0:
        print(f"\n总token数: {token_count}")
        print(f"总熵: {H_total}")
        print(f"平均每token熵: {H_total / token_count}")
        print(f"模型存储的信息: {model_stored_info}")
        print(f"剩余信息熵: {remaining_info}")
    
    # 保存结果（只在主进程上）
    if torch.distributed.get_rank() == 0:
        results = {
            'model_stored_info': model_stored_info,
            'remaining_info': remaining_info,
            'total_entropy': H_total,
            'token_count': token_count,
            'average_entropy_per_token': H_total / token_count
        }
        
        # 将结果保存到文件
        save_path = os.path.join(args.load, 'entropy_results.pt')
        print(f"保存结果到: {save_path}")
        torch.save(results, save_path)

if __name__ == "__main__":
    main()

