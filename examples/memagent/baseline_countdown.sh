torchrun \
    --nproc_per_node=8 \
    -m RL2.trainer.ppo \
    train_data.path=train@Chenmien/Countdown \
    train_data.prompts_per_rollout=128 \
    train_data.responses_per_prompt=4 \
    test_data.path=test@Chenmien/Countdown \
    actor.model_name=Qwen/Qwen3-4B-Base \
    actor.max_length_per_device=8192 \
    rollout.train_sampling_params.max_new_tokens=4096 \
    "rollout.train_sampling_params.stop=['</answer>']" \
    rollout.env_path=envs/countdown.py \
    trainer.project=MemAgent \
    trainer.experiment_name=qwen3-4b-baseline \
    trainer.test_freq=8 \
    trainer.save_freq=32