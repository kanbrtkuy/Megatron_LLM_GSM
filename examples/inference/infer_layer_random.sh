#!/bin/bash

# test the "0.2B" parameter model

#################### repeat layer setting ###############
# base layer 是12层。可选值: none, every_layer, block, alternate, custom

# 'every_layer': 每层重复模式: 1234->11223344
# 假设基础层数为12，每层重复一次，最终有24层
            
# 'block': 块重复模式: 1234->12341234
# 假设基础层数为12，整个块重复一次，最终有24层
                
# 'alternate': 交替重复模式: 如12323234

# 'custom': 自定义

REPEAT_MODE="every_layer"   ###need revise here!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!1

# 例如 [1,2,3,3,4,4,5,5,6,6,7,7,8,8,9,9,10,10,11,11,12,12]
CUSTOM_REPEAT_PATTERN="[1,1,2,2,3,3,4,4]"

# 导出环境变量
export REPEAT_MODE=$REPEAT_MODE
if [ "$REPEAT_MODE" == "custom" ]; then
    export REPEAT_PATTERN=$CUSTOM_REPEAT_PATTERN
fi

# 记录配置
echo "Running with layer repetition mode: $REPEAT_MODE" #> ./layer_repeat_config.log
[ "$REPEAT_MODE" == "custom" ] && echo "Custom repeat pattern: $CUSTOM_REPEAT_PATTERN" #>> ./layer_repeat_config.log
####################



export CUDA_DEVICE_MAX_CONNECTIONS=1
export CUDA_LAUNCH_BLOCKING=1
export TORCH_USE_CUDA_DSA=1

pip install flask-restful
pip install tqdm  # 用于进度显示

DISTRIBUTED_ARGS="--nproc_per_node 1 \
                 --nnodes 1 \
                 --node_rank 0 \
                 --master_addr localhost \
                 --master_port 6000"
                 
CHECKPOINT=./results/layer_p50_b4_v4/ckpts/  #./results/train_base_random/ckpts/
VOCAB_FILE=./datasets/gpt2-vocab.json
MERGE_FILE=./datasets/gpt2-merges.txt
DATA_PATH=./datasets/prefix_text_document/randomx3_text_document  # 确保这是训练数据的路径


# 运行脚本,micro-batch-size需要改！！
torchrun $DISTRIBUTED_ARGS tools/run_logits_random.py \
   --tensor-model-parallel-size 1 \
   --pipeline-model-parallel-size 1 \
   --num-layers 4 \
   --hidden-size 96 \
   --load ${CHECKPOINT} \
   --num-attention-heads 4 \
   --max-position-embeddings 1024 \
   --tokenizer-type RandomNumberTokenizer \
   --fp16 \
   --micro-batch-size 48 \
   --seq-length 1024 \
   --vocab-file $VOCAB_FILE \
   --merge-file $MERGE_FILE \
   --seed 1234 \
   --data-path ${DATA_PATH} \
   --split 1000,0,0 