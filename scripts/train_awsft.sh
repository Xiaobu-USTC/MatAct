#!/usr/bin/env bash
# Stage 1: action-weighted SFT (Hugging Face Transformers + DeepSpeed ZeRO-3, 8 GPUs).
# Global batch 64 = 8 GPUs x 1 sample x 8 accumulation steps (Supplementary Table 6).
# Self-training rounds reuse this script with TRAIN_FILE=data/self_train/roundK/train_messages.jsonl,
# EPOCHS=1, LR=5e-7 and MODEL set to the previous checkpoint.
set -euo pipefail
cd "$(dirname "$0")/.."
MODEL=${MODEL:-Qwen/Qwen3-8B}
OUT=${OUT:-outputs/qwen3-8b-awsft}
TRAIN_FILE=${TRAIN_FILE:-data/sft/train_messages.jsonl}
VALID_FILE=${VALID_FILE:-data/sft/valid_messages.jsonl}
EPOCHS=${EPOCHS:-3}
LR=${LR:-1e-5}
SEED=${SEED:-42}   # the paper uses three runs: 13, 42, 3407

torchrun --nproc_per_node "${NPROC:-8}" -m matact.train.weighted_sft \
  --model "$MODEL" \
  --train-file "$TRAIN_FILE" \
  --valid-file "$VALID_FILE" \
  --output-dir "$OUT" \
  --epochs "$EPOCHS" --learning-rate "$LR" --warmup-ratio 0.03 --weight-decay 0.1 \
  --per-device-batch-size 1 --gradient-accumulation-steps 8 --max-length 4900 \
  --bf16 --gradient-checkpointing --deepspeed configs/ds_zero3.json --seed "$SEED" "$@"
