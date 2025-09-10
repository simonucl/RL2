torchrun \
    --nproc_per_node=4 \
    -m RL2.trainer.sft \
    data.path=simonycl/gsm8k_training_positive_1k_transformed \
    data.max_length=8192 \
    data.batch_size=64 \
    actor.model_name=meta-llama/Llama-3.2-1B-Instruct \
    actor.sp_size=4 \
    actor.max_length_per_device=4096 \
    actor.lr=1e-5 \
    trainer.project=synthetic_gsm \
    trainer.experiment_name=llama-3.2-1b-inst \
    trainer.n_epochs=4