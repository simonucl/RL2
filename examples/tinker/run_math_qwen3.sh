#!/bin/bash

# Replicates tinker-cookbook MATH training with RL2-Tinker
#
# Original command:
# python -m tinker_cookbook.recipes.math_rl.train \
#     env=math model_name="Qwen/Qwen3-8B-Base" \
#     group_size=16 groups_per_batch=64 \
#     learning_rate=1e-4 max_tokens=4096 \
#     wandb_project="tinker-examples" \
#     wandb_name="math-Qwen3-8B-Base-16group-64batch-1e-4lr-4096tokens"

# Prepare data if not exists
if [ ! -f math_train.jsonl ]; then
    echo "Preparing MATH dataset..."
    python examples/prepare_math_data.py
fi

export TINKER_API_KEY=""
export WANDB_API_KEY=""

# Run RL2-Tinker training
python -m RL2.trainer.tinker_ppo \
    tinker.model_name="Qwen/Qwen3-8B-Base" \
    tinker.lora_rank=32 \
    tinker.learning_rate=1e-4 \
    tinker.max_tokens=4096 \
    tinker.kl_penalty_coef=0.0 \
    tinker.loss_fn=importance_sampling \
    tinker.enable_async_overlap=true \
    train_data.path=math_train.jsonl \
    train_data.prompts_per_rollout=64 \
    train_data.responses_per_prompt=16 \
    test_data.path=math_test.jsonl \
    rollout.env_module=RL2/envs/orz.py \
    rollout.train_sampling_params.temperature=1.0 \
    adv.responses_per_prompt=16 \
    adv.global_norm=false \
    adv.norm_var=false \
    trainer.n_epochs=1 \
    trainer.use_wandb=true \
    trainer.save_freq=20 \
    trainer.test_freq=20 \
    wandb.project="tinker-examples" \
    wandb.name="math-Qwen3-8B-Base-16group-64batch-1e-4lr-4096tokens-rl2"
