MODEL_NAME_OR_PATH=model_dir/chatglm/
PTUNING_CHECKPOINT=model_dir/checkpoint-2500/
PRE_SEQ_LEN=64
QUANTIZE=True

CUDA_VISIBLE_DEVICES=0 python server.py \
    --model_name_or_path $MODEL_NAME_OR_PATH \
    --ptuning_checkpoint $PTUNING_CHECKPOINT \
    --pre_seq_len $PRE_SEQ_LEN \
    --quantize $QUANTIZE