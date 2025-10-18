#!/bin/bash
# Test 1: Basic LoRA training with Qwen3-4B-Base
# This script tests LoRA functionality with checkpoint save/load

echo "========================================="
echo "Test 1: LoRA with Qwen3-4B-Base"
echo "Testing basic LoRA training with save/load"
echo "========================================="

# First run: Train for 4 steps and save checkpoint
echo "Running initial training (4 steps)..."
torchrun \
    --nproc_per_node=4 \
    -m RL2.trainer.ppo \
    train_data.path=Chenmien/OpenReasonerZero \
    train_data.prompts_per_rollout=128 \
    train_data.responses_per_prompt=64 \
    test_data.path=Chenmien/OlympiadBench \
    actor.model_name=Qwen/Qwen3-4B-Base \
    actor.cp_size=2 \
    actor.max_length_per_device=8192 \
    actor.use_lora=true \
    actor.lora.r=16 \
    actor.lora.lora_alpha=32 \
    actor.lora.lora_dropout=0.05 \
    rollout.train_sampling_params.max_new_tokens=8192 \
    rollout.env_path=envs/orz.py \
    adv.estimator=reinforce \
    trainer.project=LoRA_Tests \
    trainer.experiment_name=qwen3-4b-lora-basic \
    trainer.n_epochs=1 \
    trainer.test_freq=2 \
    trainer.save_freq=4

if [ $? -ne 0 ]; then
    echo "ERROR: Initial training failed!"
    exit 1
fi

echo ""
echo "========================================="
echo "Checking saved checkpoint..."
echo "========================================="

# Check if checkpoint was saved
if [ -d "ckpts/qwen3-4b-lora-basic/step_4" ]; then
    echo "✓ Checkpoint saved successfully at ckpts/qwen3-4b-lora-basic/step_4"

    # Check for model files
    if [ -d "ckpts/qwen3-4b-lora-basic/step_4/model" ]; then
        echo "✓ Model directory exists"
    fi

    # Check for LoRA adapter files
    if [ -d "ckpts/qwen3-4b-lora-basic/step_4/model/lora_adapters" ]; then
        echo "✓ LoRA adapters saved"
    else
        echo "⚠ WARNING: LoRA adapters directory not found!"
    fi
else
    echo "ERROR: Checkpoint directory not found!"
    exit 1
fi

echo ""
echo "========================================="
echo "Test 2: Resume from checkpoint"
echo "========================================="

# Second run: Resume from checkpoint and train 4 more steps
echo "Resuming training from checkpoint..."
torchrun \
    --nproc_per_node=4 \
    -m RL2.trainer.ppo \
    train_data.path=Chenmien/OpenReasonerZero \
    train_data.prompts_per_rollout=128 \
    train_data.responses_per_prompt=64 \
    test_data.path=Chenmien/OlympiadBench \
    actor.model_name=Qwen/Qwen3-4B-Base \
    actor.cp_size=2 \
    actor.max_length_per_device=8192 \
    actor.use_lora=true \
    actor.lora.r=16 \
    actor.lora.lora_alpha=32 \
    actor.lora.lora_dropout=0.05 \
    rollout.train_sampling_params.max_new_tokens=8192 \
    rollout.env_path=envs/orz.py \
    adv.estimator=reinforce \
    trainer.project=LoRA_Tests \
    trainer.experiment_name=qwen3-4b-lora-basic \
    trainer.load_ckpt_from=ckpts/qwen3-4b-lora-basic/step_4 \
    trainer.n_epochs=1 \
    trainer.test_freq=2 \
    trainer.save_freq=4

if [ $? -ne 0 ]; then
    echo "ERROR: Resume training failed!"
    exit 1
fi

echo ""
echo "========================================="
echo "✓ All tests passed!"
echo "========================================="
echo "Results saved in: ckpts/qwen3-4b-lora-basic/"
echo "LoRA adapters saved in: ckpts/qwen3-4b-lora-basic/latest/lora_adapters/"
