MODEL_NAME_OR_PATH=./
PTUNING_CHECKPOINT=./
PRE_SEQ_LEN=128

CUDA_VISIBLE_DEVICES=0 python server.py \
    --model_name_or_path $MODEL_NAME_OR_PATH \
    --ptuning_checkpoint $PTUNING_CHECKPOINT \
    --pre_seq_len $PRE_SEQ_LEN