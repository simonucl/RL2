export SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1

torchrun \
    --nproc_per_node=8 \
    -m RL2.trainer.ppo \
    train_data.path=simonycl/math-12k \
    train_data.prompts_per_rollout=128 \
    train_data.responses_per_prompt=8 \
    test_data.path=Chenmien/OlympiadBench \
    actor.model_name=simonycl/OLMoE-1B-7B-0125 \
    actor.cp_size=2 \
    actor.max_length_per_device=4096 \
    +actor.track_tis=true \
    rollout.train_sampling_params.max_new_tokens=3072 \
    rollout.env_path=envs/orz.py \
    +rollout.server_args.context_length=8192 \
    adv.estimator=reinforce \
    trainer.project=OpenMoERL \
    trainer.experiment_name=olmoe-1b-7b-0125-base \
    trainer.test_freq=8 \
    trainer.save_freq=32 \
    trainer.n_epochs=20