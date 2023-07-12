
model_name_or_path
HOST="0.0.0.0"
POST=6666
MODEL_NAME_OR_PATH=""
PTUNING_CHECKPOINT=""
PRE_SEQ_LEN=128

CUDA_VISIBLE_DEVICES=0 python3 web.py \
    --model_name_or_path "./model_dir/thudm/chatglm-6b/" \
    --ptuning_checkpoint "./output2/output/sim-chatglm-6b-pt-128-2e-2/checkpoint-1800" \
    --pre_seq_len $PRE_SEQ_LEN