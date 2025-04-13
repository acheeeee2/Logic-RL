#!/bin/bash
set -x

# 设置模型路径 - 使用开源可用的模型
MODEL_PATH="gpt2"

# MacOS环境设置
export VLLM_BACKEND=LOGITS_ONLY
export PYTHONPATH=$PYTHONPATH:$(pwd)

# 使用样本数据
DATA_DIR="data/kk/instruct/3ppl"
TRAIN_FILE="$DATA_DIR/train.parquet"
VAL_FILE="$DATA_DIR/test.parquet"

python -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=$TRAIN_FILE \
    data.val_files=$VAL_FILE \
    data.train_batch_size=4 \
    data.val_batch_size=2 \
    data.max_prompt_length=400 \
    data.max_response_length=1024 \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.actor.optim.lr=3e-7 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=2 \
    actor_rollout_ref.actor.ppo_micro_batch_size=1 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.model.enable_gradient_checkpointing=False \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.grad_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size=1 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=hf \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.n=2 \
    actor_rollout_ref.ref.log_prob_micro_batch_size=1 \
    actor_rollout_ref.ref.fsdp_config.param_offload=False \
    algorithm.kl_ctrl.kl_coef=0.001 \
    trainer.critic_warmup=0 \
    trainer.logger=['tensorboard'] \
    trainer.project_name='GRPO_logic_KK' \
    trainer.experiment_name='GPT2-MacOS-Test' \
    trainer.n_gpus_per_node=0 \
    trainer.nnodes=1 \
    trainer.default_local_dir=./output \
    trainer.default_hdfs_dir=null \
    trainer.save_freq=5 \
    trainer.test_freq=5 \
    trainer.total_epochs=1 $@ 2>&1 | tee grpo_macos.log 