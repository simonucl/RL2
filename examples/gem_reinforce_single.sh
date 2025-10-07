torchrun \
    --nproc_per_node=8 \
    -m RL2.trainer.ppo \
    train_data.prompts_per_rollout=64 \
    test_data.prompts_per_rollout=64 \
    actor.model_name=Qwen/Qwen3-4B-Base \
    actor.max_length_per_device=8192 \
    rollout.train_sampling_params.max_new_tokens=4096 \
    rollout.env_path=envs/sudoku.py \
    rollout.max_turns=30 \
    adv.global_norm=true \
    adv.norm_var=true \
    adv.estimator=reinforce \
    trainer.project=GEM \
    trainer.experiment_name=sudoku_qwen3-4b_reinforce \
    trainer.n_epochs=200 \
    trainer.save_freq=64