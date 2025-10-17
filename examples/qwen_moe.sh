# hf download qwen/Qwen3-4B-Base --local-dir /root/Qwen3-4B-Base
# hf download qwen/Qwen3-30B-A3B-Base --local-dir /root/Qwen3-30B-A3B-Base

torchrun \
    --nproc_per_node=8 \
    -m RL2.trainer.ppo \
    actor=megatron \
    ref_actor=megatron \
    critic=megatron \
    train_data.path=Chenmien/OpenReasonerZero \
    train_data.prompts_per_rollout=16 \
    train_data.responses_per_prompt=4 \
    test_data.path=Chenmien/OlympiadBench \
    actor.model_name=/root/Qwen3-30B-A3B-Base \
    actor.pp_size=1 \
    actor.cp_size=1 \
    actor.ep_size=2 \
    actor.tp_size=4 \
    actor.max_length_per_device=8192 \
    actor.optimizer.lr=5e-7 \
    rollout.server_args.tp_size=2 \
    rollout.server_args.mem_fraction_static=0.85 \
    rollout.train_sampling_params.max_new_tokens=5000 \
    rollout.env_path=envs/orz.py \
    adv.estimator=reinforce \
    trainer.project=QwenMoE \
    trainer.experiment_name=qwen3-30b-a3b-base \
    trainer.test_freq=8 \
    trainer.save_freq=32
