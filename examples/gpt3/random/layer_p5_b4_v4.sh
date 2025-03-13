#!/bin/bash

# Runs the "0.2B" parameter model

#################### repeat layer setting ###############
# base layer 是12层。可选值: none, every_layer, block, alternate, custom

# 'every_layer': 每层重复模式: 1234->11223344
# 假设基础层数为12，每层重复一次，最终有24层
            
# 'block': 块重复模式: 1234->12341234
# 假设基础层数为12，整个块重复一次，最终有24层
                
# 'alternate': 交替重复模式: 如12323234

# 'custom': 自定义

REPEAT_MODE="every_layer"

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
GPUS_PER_NODE=8
# Change for multinode config
MASTER_ADDR=localhost
MASTER_PORT=6000
NUM_NODES=1
NODE_RANK=0
WORLD_SIZE=$(($GPUS_PER_NODE*$NUM_NODES))

CHECKPOINT_PATH='./results/random_layer_p50_b4_v4/ckpts/'   #$1 #<Specify path>  
TENSORBOARD_LOGS_PATH='./results/random_layer_p50_b4_v4/tensorboard/' #$2 #<Specify path>
VOCAB_FILE='./datasets/gpt2-vocab.json' #$3 #<Specify path to file>/gpt2-vocab.json
MERGE_FILE='./datasets/gpt2-merges.txt' #$4 #<Specify path to file>/gpt2-merges.txt
DATA_PATH='./datasets/prefix_text_document/randomx3_text_document'  #$5 #<Specify path and file prefix>_text_document
RANDOM_FLAG='False'
# TOKENIZER_TYPE='RandomNumberTokenizer'   #'GPT2BPETokenizer'


DISTRIBUTED_ARGS=(
    --nproc_per_node $GPUS_PER_NODE 
    --nnodes $NUM_NODES 
    --master_addr $MASTER_ADDR 
    --master_port $MASTER_PORT
)

#revise
GPT_MODEL_ARGS=(
    --num-layers 4 #48   #96   #
    --hidden-size 96   #192 #1920  #128    #12288  #
    --num-attention-heads 4    #96  #
    --seq-length 1024   #2048 
    --max-position-embeddings 1024  #2048 
    --attention-backend auto # Can use (flash/fused/unfused/local)
)

TRAINING_ARGS=(
    --micro-batch-size 96 #48  #96 
    --global-batch-size 768  #384 #  #192  #1536     #revise 
    # --rampup-batch-size 16 16 5859375   #revise
    --train-iters 8140  #8140   #833333 #640000  #52084  #maybe can be larger?
    --weight-decay 0.02
    --adam-beta1 0.9 
    --adam-beta2 0.95 
    --init-method-std 0.006 
    --clip-grad 1.0 
    --fp16
    --lr 0.002
    --lr-decay-style cosine 
    --min-lr 0.0001
    --lr-warmup-fraction 0.01919975424  #0.01
    --lr-decay-iters 24000
)

#revise
MODEL_PARALLEL_ARGS=(
	--tensor-model-parallel-size 1  #8 
	--pipeline-model-parallel-size 1  #16 
)

DATA_ARGS=(
    --data-path $DATA_PATH 
    --vocab-file $VOCAB_FILE 
    --merge-file $MERGE_FILE 
    --split 1000,0,0    #949,50,1   ##??
)

EVAL_AND_LOGGING_ARGS=(
    --log-interval 200 #400
    --save-interval 800   #10000
    --eval-interval 999999999999999999999 
    --save $CHECKPOINT_PATH 
    --load $CHECKPOINT_PATH 
    --eval-iters 9999999999999999
    --tensorboard-dir $TENSORBOARD_LOGS_PATH 
)


torchrun ${DISTRIBUTED_ARGS[@]} pretrain_gpt.py \
    ${GPT_MODEL_ARGS[@]} \
    ${TRAINING_ARGS[@]} \
    ${MODEL_PARALLEL_ARGS[@]} \
    ${DATA_ARGS[@]} \
    ${EVAL_AND_LOGGING_ARGS[@]}  > ./results/random_layer_p50_b4_v4.log
