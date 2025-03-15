#!/bin/bash

# test the "0.2B" parameter model

#################### repeat layer setting ###############
##TODO
REPEAT_MODE="none"   ###need revise here!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!1

# 例如 [1,2,3,3,4,4,5,5,6,6,7,7,8,8,9,9,10,10,11,11,12,12]
CUSTOM_REPEAT_PATTERN="[1,1,2,2,3,3,4,4]"

##TODO
# Start iteration setting
START_ITER=60000  # Set your start iteration number here

# 导出环境变量
export REPEAT_MODE=$REPEAT_MODE
if [ "$REPEAT_MODE" == "custom" ]; then
   export REPEAT_PATTERN=$CUSTOM_REPEAT_PATTERN
fi

# 记录配置
echo "Running with layer repetition mode: $REPEAT_MODE" 
[ "$REPEAT_MODE" == "custom" ] && echo "Custom repeat pattern: $CUSTOM_REPEAT_PATTERN"
echo "Starting from iteration: $START_ITER"
####################

OUTPUT_PATH="./datasets/igsm/layer_le15_pred.txt"
export OUTPUT_PATH=$OUTPUT_PATH

export CUDA_DEVICE_MAX_CONNECTIONS=1
export CUDA_LAUNCH_BLOCKING=1
export TORCH_USE_CUDA_DSA=1

pip install flask-restful
pip install tqdm

DISTRIBUTED_ARGS="--nproc_per_node 1 \
                --nnodes 1 \
                --node_rank 0 \
                --master_addr localhost \
                --master_port 6000"

##TODO
CHECKPOINT_BASE="./results/gsm_base_p50_b4_v0/ckpts"

DIR_NAME=$(basename $(dirname ${CHECKPOINT_BASE}))
LOG_FILE="./results/${DIR_NAME}_inference_log.txt"

VOCAB_FILE=./datasets/gpt2-vocab.json
MERGE_FILE=./datasets/gpt2-merges.txt
DATA_PATH=./datasets/prefix_text_document/gsm_text_document

touch $LOG_FILE

echo "Starting inference with minimum iteration: $START_ITER" | tee -a $LOG_FILE
echo "----------------------------------------" | tee -a $LOG_FILE

# 遍历所有迭代的checkpoint目录
for iter_dir in ${CHECKPOINT_BASE}/iter_*; do
  if [ -d "$iter_dir" ]; then
      iter_num=$(basename $iter_dir | sed 's/iter_//')
      
      # 检查是否大于起始迭代
      if [ "$iter_num" -gt "$START_ITER" ]; then
          echo "Running inference for iteration ${iter_num}..." | tee -a $LOG_FILE
          
          # 更新latest_checkpointed_iteration.txt
          echo $iter_num > ${CHECKPOINT_BASE}/latest_checkpointed_iteration.txt
          
          # 禁用tqdm进度条
          export TQDM_DISABLE=1
          
          # ##TODO
          torchrun $DISTRIBUTED_ARGS tools/run_logits_gsm.py \
           --tensor-model-parallel-size 1 \
           --pipeline-model-parallel-size 1 \
           --num-layers 4 \
           --hidden-size 768 \
           --load ${CHECKPOINT_BASE} \
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
           --split 1000,0,0  2>&1 | tee -a $LOG_FILE
          
          sleep 5
          
          echo "Completed inference for iteration ${iter_num}" | tee -a $LOG_FILE
          echo "----------------------------------------" | tee -a $LOG_FILE
      else
          echo "Skipping iteration ${iter_num} (less than or equal to ${START_ITER})" | tee -a $LOG_FILE
      fi
  fi
done

echo "All inference runs completed!" | tee -a $LOG_FILE