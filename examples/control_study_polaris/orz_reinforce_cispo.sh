export WEAVE_PRINT_CALL_LINK=false
pkill -9 -f "RL2.trainer.ppo"
pkill -9 -f "sglang"
sleep 3

torchrun \
    --nproc_per_node=8 \
    -m RL2.trainer.ppo \
    train_data.path=Chenmien/OpenReasonerZero \
    train_data.prompts_per_rollout=64 \
    train_data.responses_per_prompt=8 \
    test_data.path=Chenmien/OlympiadBench \
    actor.model_name=Qwen/Qwen3-4B-Base \
    actor.cp_size=2 \
    actor.max_length_per_device=8192 \
    actor.avg_level=token \
    actor.loss_type=cispo \
    adv.estimator=reinforce \
    adv.norm_var=true \
    rollout.train_sampling_params.max_new_tokens=8192 \
    rollout.env_path=envs/orz.py \
    trainer.project=OpenReasonerZero \
    trainer.experiment_name=qwen3-4b-base-cispo \
    trainer.test_freq=8 \
    trainer.save_freq=32 \
    trainer.load_ckpt_from=latest