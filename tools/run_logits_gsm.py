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



data_num = 100

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


def autoregressive_generate(
    model, 
    tokenizer,
    input_tokens: torch.Tensor,  # 初始输入张量 [1, seq_len]
    max_new_tokens: int = 100,
    temperature: float = 1.0,
    top_k: int = 50,
    eod_token_id: int = None
):
    """
    自回归生成函数
    Args:
        input_tokens: 初始输入token序列 (batch_size=1)
        max_new_tokens: 最大生成token数
        temperature: 温度参数(>1更随机，<1更确定)
        top_k: top-k采样参数
        eod_token_id: 终止token ID
    """
    model.eval()
    generated_sequence = input_tokens.clone()
    is_finished = torch.zeros(1, dtype=torch.bool, device='cuda')

    # 预分配注意力掩码空间 (优化长序列性能)
    max_length = input_tokens.size(1) + max_new_tokens
    causal_mask = torch.tril(torch.ones(
        1, max_length, max_length, 
        dtype=torch.bool, 
        device='cuda'
    ))

    for step in range(max_new_tokens):
        if is_finished.all():
            break

        # 当前序列长度
        curr_seq_len = generated_sequence.size(1)
        
        # 准备输入参数
        attention_mask = causal_mask[:, :curr_seq_len, :curr_seq_len]
        position_ids = torch.arange(curr_seq_len, device='cuda').unsqueeze(0)

        # 模型前向
        with torch.no_grad():
            logits = model(input_ids=generated_sequence,position_ids=position_ids,attention_mask=attention_mask)[:, -1, :]  # 只取最后一个位置的logits [1, vocab_size]

        # 采样策略
        logits = logits / temperature
        if top_k > 0:
            top_values = torch.topk(logits, top_k).values
            logits[logits < top_values[:, -1:]] = -float('Inf')
        
        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)

        # 更新序列
        generated_sequence = torch.cat([generated_sequence, next_token], dim=1)
        
        # 检查终止条件
        is_finished = (next_token == eod_token_id) | is_finished

    # 后处理
    # import pdb;pdb.set_trace()
    generated_ids = generated_sequence[0].cpu().tolist()
    if eod_token_id in generated_ids:
        eod_pos = generated_ids.index(eod_token_id)
        generated_ids = generated_ids[:eod_pos+1]
    
    return tokenizer.detokenize(generated_ids)

from pathlib import Path
def load_questions(input_file: str):
    """从文本文件加载问题列表"""
    path = Path(input_file)
    if not path.exists():
        raise FileNotFoundError(f"输入文件 {input_file} 不存在")
        
    with path.open('r', encoding='utf-8') as f:
        questions = [line.strip() for line in f if line.strip()]
        
    return questions


################
from difflib import SequenceMatcher
import re

def extract_solution_answer(text):
    """提取Solution和Answer部分（保持原有逻辑不变）"""
    match = re.search(r'(Solution:.+?)(?=$|\n|Question:)', text, re.DOTALL)
    return match.group(1).strip() if match else ""

def calculate_similarity(text1, text2):
    """计算两个文本的相似度（保持原有逻辑不变）"""
    return SequenceMatcher(None, text1, text2).ratio()

import re

def extract_answers_from_first_file(file_path):
    """
    从第一个格式的文本文件中提取所有"Answer:"后面的数值
    
    参数:
        file_path (str): 文本文件的路径
        
    返回:
        list: 包含所有提取出的数值的列表
    """
    # 创建一个空列表用于存储提取的答案
    answers = []
    
    try:
        # 打开并读取文件内容
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
        
        # 使用正则表达式匹配"Answer:"后面的数字
        answer_pattern = re.compile(r'Answer:\s*(\d+)')
        
        # 查找所有匹配项
        matches = answer_pattern.finditer(content)
        
        # 将找到的每个数字转换为整数并添加到列表中
        for match in matches:
            number = int(match.group(1))
            answers.append(number)
            
        print(f"成功从第一个文件中提取了{len(answers)}个答案")
        
    except FileNotFoundError:
        print(f"错误: 找不到文件 '{file_path}'")
    except Exception as e:
        print(f"处理第一个文件时发生错误: {e}")
    
    return answers

def extract_answer_number(text):
    """
    Extracts the number following "Answer: " in the text.
    Returns 999 if no valid answer is found.
    
    Args:
        text (str): The prediction text to parse
    
    Returns:
        int: The extracted answer number or 999 if extraction fails
    """
    import re
    
    # Look for "Answer: X" pattern
    answer_match = re.search(r'Answer:\s*(\d+)(?:\s*$|\s*\n)', text)
    
    if answer_match:
        # Valid answer found
        return int(answer_match.group(1))
    else:
        # No valid answer found
        return 999


def process_predictions(file_path):
    """
    Process all predictions in the file, splitting by "########"
    
    Args:
        prediction_text (str): The full prediction text
    
    Returns:
        list: List of extracted answers (integers)
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        prediction_text = f.read()

    predictions = prediction_text.split("######## ")
    # import pdb;pdb.set_trace()
    results = []
    
    for pred in predictions:
        if pred.strip():  # Skip empty predictions
            answer = extract_answer_number(pred)
            results.append(answer)
    
    return results


def process_predictions_gt(file_path):
    import re
    results = []
    
    # Read the file line by line
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            # Look for "Answer: X" pattern in each line
            answer_match = re.search(r'Answer:\s*(\d+)(?:\s*$|\s*\n)', line)
            
            if answer_match:
                # Valid answer found
                results.append(int(answer_match.group(1)))
    
    return results


def evaluate_outputs(gt_file, pred_file):
    # 读取文件（增加编码处理）
    gt_lines = process_predictions_gt(gt_file)[:data_num]
    pred_lines = process_predictions(pred_file)
    # import pdb;pdb.set_trace()
    
    # 初始化统计指标
    metrics = {
        'total': len(pred_lines),
        'correct': 0,
        'similarities': []
    }
    
    # 逐对评估
    for idx in range(len(pred_lines)):
        gt_answer = gt_lines[idx]
        pred_answer = pred_lines[idx]
        # 统计正确率
        if gt_answer == pred_answer:
            metrics['correct'] += 1
    
    print(gt_lines,  pred_lines)
    print(len(gt_lines),len(pred_lines))
    # 计算综合指标
    metrics.update({
        'accuracy': metrics['correct'] / metrics['total']
    })
    
    return metrics




def main():

    # STEP 1 - 初始化模型并设置默认参数
    initialize_megatron(args_defaults={
        'no_load_rng': True,
        'no_load_optim': True,
        'micro_batch_size': 1
    })
    args = get_args()
    tokenizer = get_tokenizer()

    questions = load_questions("./datasets/igsm/igsm_med_pq_eval_le15_questions.txt")  ##revise
    label = "./datasets/igsm/igsm_med_pq_eval_le15_full_texts.txt" # "./datasets/igsm/igsm_med_pq_eval_e23_full_texts.txt"
    output_file = os.environ.get('OUTPUT_PATH')
    # import pdb;pdb.set_trace()

    
    # STEP 2 - 加载模型
    model = get_model(model_provider, wrap_with_ddp=False)
    load_checkpoint(model, None, None)
    model = model[0]  # 获取实际模型
    model.eval()  # 设置为评估模式

    generation_config = {
        'max_new_tokens': 1280,
        'temperature': 1.0,
        'top_k': 50,
        'eod_token_id': tokenizer.eod
    }
    
    
    # 计算模型熵
    H_total = 0.0
    token_count = 0
    
    with open(output_file, 'w') as f:
        for batch_idx, batch in enumerate(tqdm(questions[:data_num])):  #revise here
            # 获取输入数据
            #######################
            tokens = tokenizer.tokenize(batch)
            tokens = torch.tensor([tokens]).cuda().int()
            attention_mask = torch.ones_like(tokens, dtype=torch.bool, device=tokens.device)
            # print(tokens.device)
            
            # 准备位置ID
            batch_size, seq_length = tokens.size()
            position_ids = torch.arange(seq_length, dtype=torch.long, device=tokens.device)
            position_ids = position_ids.unsqueeze(0).expand_as(tokens)
            
            # 获取模型输出logits seq
            generated_text = autoregressive_generate(model=model,tokenizer=tokenizer,input_tokens=tokens,**generation_config) 
            
            if "Solution:" not in generated_text or "Answer:" not in generated_text:
                print(f"Warning: Generated text missing Solution/Answer for batch {batch_idx}")
            
            # 写入文件并立即刷新
            # print(generated_text)
            generated_text = generated_text.replace("<|endoftext|>", "").strip()
            f.write(generated_text + "\n")
            f.write("######## \n")
            f.flush()


    results = evaluate_outputs(label, output_file)
    print(f"acc: {results['accuracy']:.2%}")
    




if __name__ == "__main__":
    main()

