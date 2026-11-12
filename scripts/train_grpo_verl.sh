#!/usr/bin/env bash
# Stage 2: verifiable, cost-aware GRPO with verl multi-turn tool rollouts (8 GPUs).
# Rewards come from matact/integrations/verl_reward.py, which replays the policy's actions
# in the MatAct environment; tools come from matact/integrations/verl_tools.py.
#   REWARD_FN=compute_score_generic   generic tool-RL baseline (format, execution, final answer only)
#   LR=5e-7 MODEL=<self-trained ckpt> the policy-optimization step of a self-training round
# Key names follow recent verl releases; check them against your installed version.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
MODEL=${MODEL:-outputs/qwen3-8b-awsft}
LR=${LR:-1e-6}
REWARD_FN=${REWARD_FN:-compute_score}
EXP_NAME=${EXP_NAME:-qwen3-8b-grpo}
python3 scripts/to_parquet.py data/rl/train.jsonl data/rl/valid.jsonl

python3 -m verl.trainer.main_ppo \
  algorithm.adv_estimator=grpo \
  algorithm.use_kl_in_reward=False \
  data.train_files=data/rl/train.parquet \
  data.val_files=data/rl/valid.parquet \
  data.train_batch_size=64 \
  data.max_prompt_length=3072 \
  data.max_response_length=4900 \
  data.return_raw_chat=True \
  actor_rollout_ref.model.path="$MODEL" \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.actor.optim.lr="$LR" \
  actor_rollout_ref.actor.ppo_mini_batch_size=64 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.actor.clip_ratio=0.2 \
  actor_rollout_ref.actor.use_kl_loss=True \
  actor_rollout_ref.actor.kl_loss_coef=0.02 \
  actor_rollout_ref.actor.kl_loss_type=low_var_kl \
  actor_rollout_ref.actor.fsdp_config.param_offload=False \
  actor_rollout_ref.rollout.name=sglang \
  actor_rollout_ref.rollout.n=8 \
  actor_rollout_ref.rollout.temperature=1.0 \
  actor_rollout_ref.rollout.top_p=1.0 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.rollout.multi_turn.enable=True \
  actor_rollout_ref.rollout.multi_turn.max_assistant_turns=16 \
  actor_rollout_ref.rollout.multi_turn.tool_config_path=data/rl/verl_tools.yaml \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
  custom_reward_function.path=matact/integrations/verl_reward.py \
  custom_reward_function.name="$REWARD_FN" \
  trainer.n_gpus_per_node="${NPROC:-8}" \
  trainer.nnodes=1 \
  trainer.total_epochs=1 \
  trainer.project_name=matact \
  trainer.experiment_name="$EXP_NAME" \
  trainer.logger='["console"]' "$@"
