#!/bin/bash

# test the "0.2B" parameter model

#################### repeat layer setting ###############

REPEAT_MODE="none"   ###need revise here!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!1


CUSTOM_REPEAT_PATTERN="[1,2,3,4]"

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

OUTPUT_PATH="./datasets/igsm/base_le15_pred.txt"
export OUTPUT_PATH=$OUTPUT_PATH

CHECKPOINT=./results/gsm_base_p100_b8_v0_ver1_backup/ckpts/  #./results/train_base_random/ckpts/
VOCAB_FILE=./datasets/gpt2-vocab.json
MERGE_FILE=./datasets/gpt2-merges.txt
DATA_PATH=./datasets/prefix_text_document/gsm_text_document  # 确保这是训练数据的路径



# 运行脚本,micro-batch-size需要改！！
torchrun $DISTRIBUTED_ARGS tools/run_logits_gsm.py \
   --tensor-model-parallel-size 1 \
   --pipeline-model-parallel-size 1 \
   --num-layers 8 \
   --hidden-size 768 \
   --load ${CHECKPOINT} \
   --num-attention-heads 12 \
   --max-position-embeddings 2048 \
   --tokenizer-type GPT2BPETokenizer \
   --fp16 \
   --micro-batch-size 1 \
   --seq-length 2048 \
   --vocab-file $VOCAB_FILE \
   --merge-file $MERGE_FILE \
   --seed 1234 \
   --data-path ${DATA_PATH} \
   --split 1000,0,0 