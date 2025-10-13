export SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1
torchrun \
    --nproc_per_node=8 \
    -m RL2.trainer.ppo \
    train_data.path=simonycl/math-12k \
    train_data.prompts_per_rollout=128 \
    train_data.responses_per_prompt=8 \
    test_data.path=Chenmien/OlympiadBench \
    actor.model_name=openai/gpt-oss-20b \
    actor.max_length_per_device=8192 \
    actor.sp_size=2 \
    +actor.track_tis=true \
    +train_data.add_boxed_prompt=true \
    rollout.train_sampling_params.max_new_tokens=8192 \
    rollout.env_path=envs/orz.py \
    adv.estimator=reinforce \
    trainer.project=OpenMoERL \
    trainer.experiment_name=gpt-oss-20b-base \
    trainer.test_freq=8 \
    trainer.save_freq=32 \
    trainer.n_epochs=20