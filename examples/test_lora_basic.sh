torchrun \
    --nproc_per_node=4 \
    -m RL2.trainer.ppo \
    train_data.path=Chenmien/OpenReasonerZero \
    train_data.prompts_per_rollout=8 \
    train_data.responses_per_prompt=8 \
    test_data.path=Chenmien/OlympiadBench \
    actor.model_name=Qwen/Qwen3-4B-Base \
    actor.max_length_per_device=8192 \
    actor.use_lora=true \
    actor.lora.r=8 \
    actor.lora.lora_alpha=16 \
    actor.lora.lora_dropout=0.05 \
    rollout.train_sampling_params.max_new_tokens=400 \
    rollout.env_path=envs/orz.py \
    adv.estimator=reinforce \
    trainer.project=LoRA_Tests \
    trainer.experiment_name=qwen3-4b-lora-basic \
    trainer.n_epochs=1 \
    trainer.test_freq=10 \
    trainer.save_freq=4