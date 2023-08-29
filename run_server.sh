MODEL_NAME_OR_PATH=model_dir/chatglm/
EXTRA_MODEL_NAME_OR_PATH=model_dir/extra_model/
MODEL_TYPE=pt
INFERENCE_MODE=origin
PRE_SEQ_LEN=64
QUANTIZE=True

CUDA_VISIBLE_DEVICES=0 python server.py \
    --model_name_or_path $MODEL_NAME_OR_PATH \
    --extra_model_name_or_path $EXTRA_MODEL_NAME_OR_PATH \
    --model_type $MODEL_TYPE \
    --inference_mode $INFERENCE_MODE \
    --pre_seq_len $PRE_SEQ_LEN \
    --quantize $QUANTIZE