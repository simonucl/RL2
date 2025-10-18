torchrun \
    --nproc_per_node=8 \
    -m RL2.trainer.ppo \
    train_data.path=Chenmien/OpenReasonerZero \
    train_data.prompts_per_rollout=128 \
    train_data.responses_per_prompt=64 \
    test_data.path=Chenmien/OlympiadBench \
    actor.model_name=Qwen/Qwen3-30B-A3B-Base \
    actor.cp_size=2 \
    actor.tp_size=4 \
    actor.max_length_per_device=8192 \
    rollout.train_sampling_params.max_new_tokens=5000 \
    rollout.env_path=envs/orz.py \
    adv.estimator=reinforce \
    trainer.project=OpenReasonerZero \
    trainer.experiment_name=qwen3-30b-a3b-base-moe-fsdp \
    trainer.test_freq=8 \
    trainer.save_freq=32