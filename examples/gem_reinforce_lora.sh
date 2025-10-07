torchrun \
    --nproc_per_node=8 \
    -m RL2.trainer.ppo \
    train_data.prompts_per_rollout=64 \
    test_data.prompts_per_rollout=64 \
    actor.model_name=Qwen/Qwen3-4B-Base \
    actor.max_length_per_device=8192 \
    actor.lr=1e-4 \
    actor.tp_size=1 \
    actor.lora.use_lora=true \
    actor.lora.r=16 \
    actor.lora.lora_alpha=32 \
    actor.lora.lora_dropout=0.05 \
    rollout.train_sampling_params.max_new_tokens=4096 \
    rollout.env_path=envs/gem.py \
    rollout.max_turns=30 \
    adv.global_norm=true \
    adv.norm_var=true \
    adv.estimator=reinforce \
    trainer.project=GEM \
    trainer.experiment_name=multi-env_qwen3-4b_reinforce_lora \
    trainer.n_epochs=200 \
    trainer.save_freq=64
