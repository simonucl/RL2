torchrun \
    --nproc_per_node=4 \
    -m RL2.trainer.ppo \
    train_data.prompts_per_rollout=64 \
    test_data.prompts_per_rollout=64 \
    actor.model_name=Qwen/Qwen3-1.7B-Base \
    actor.max_length_per_device=8192 \
    actor.tp_size=1 \
    actor.lora.use_lora=true \
    actor.lora.r=16 \
    actor.lora.lora_alpha=32 \
    actor.lora.lora_dropout=0.05 \
    rollout.train_sampling_params.max_new_tokens=1024 \
    rollout.env_path=envs/gem.py \
    adv.global_norm=true \
    adv.norm_var=true \
    trainer.project=GEM \
    trainer.experiment_name=letter-counting_qwen3-1.7b_reinforce_lora \
    trainer.n_epochs=512 \
    trainer.save_freq=64
