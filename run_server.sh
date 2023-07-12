MODEL_NAME_OR_PATH=../model_dir/thudm/chatglm-6b/
PTUNING_CHECKPOINT=./output/chatglm-6b-pt-64-2e-2-0620/checkpoint-2500/
PRE_SEQ_LEN=64

CUDA_VISIBLE_DEVICES=0 python server.py \
    --model_name_or_path $MODEL_NAME_OR_PATH \
    --ptuning_checkpoint $PTUNING_CHECKPOINT \
    --pre_seq_len $PRE_SEQ_LEN