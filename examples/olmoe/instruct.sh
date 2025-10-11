export SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1

while true; do
    torchrun \
        --nproc_per_node=8 \
        -m RL2.trainer.ppo \
        train_data.path=simonycl/math-12k \
        train_data.prompts_per_rollout=128 \
        train_data.responses_per_prompt=8 \
        test_data.path=Chenmien/OlympiadBench \
        actor.model_name=simonycl/OLMoE-1B-7B-0125-Instruct \
        actor.max_length_per_device=4096 \
        +actor.track_tis=true \
        rollout.train_sampling_params.max_new_tokens=3072 \
        rollout.env_path=envs/orz.py \
        +rollout.context_length=8192 \
        adv.estimator=reinforce \
        trainer.project=OpenMoERL \
        trainer.experiment_name=olmoe-1b-7b-0125-instruct \
        trainer.load_ckpt_from=latest \
        trainer.test_freq=8 \
        trainer.save_freq=32 \
        trainer.n_epochs=20

    if [ $? -eq 0 ]; then
        echo "✅ Training completed"
        exit 0
    fi

    echo "❌ Training failed, cleaning up and restarting in 30s..."
    pkill -9 -f "python.*RL2.trainer.ppo"
    pkill -9 -f sglang
    sleep 30
done