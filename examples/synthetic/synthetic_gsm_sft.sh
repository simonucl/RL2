# torchrun \
#     --nproc_per_node=4 \
#     -m RL2.trainer.sft \
#     data.path=simonycl/gsm8k_training_positive_vs_cot_transformed \
#     data.max_length=8192 \
#     data.batch_size=64 \
#     actor.model_name=meta-llama/Llama-3.2-1B-Instruct \
#     actor.sp_size=2 \
#     actor.max_length_per_device=4096 \
#     actor.lr=1e-5 \
#     trainer.project=synthetic_gsm \
#     trainer.experiment_name=llama-3.2-1b-vs-cot-inst \
#     trainer.n_epochs=4

# sleep 30
# torchrun \
#     --nproc_per_node=4 \
#     -m RL2.trainer.sft \
#     data.path=simonycl/gsm8k_training_positive_direct_cot_1k_transformed \
#     data.max_length=8192 \
#     data.batch_size=64 \
#     actor.model_name=meta-llama/Llama-3.2-1B-Instruct \
#     actor.sp_size=2 \
#     actor.max_length_per_device=4096 \
#     actor.lr=1e-5 \
#     trainer.project=synthetic_gsm \
#     trainer.experiment_name=llama-3.2-1b-direct-cot-inst \
#     trainer.n_epochs=4

# sleep 30

torchrun \
    --nproc_per_node=4 \
    -m RL2.trainer.sft \
    data.path=simonycl/gsm8k_training_positive_1k_regenerated \
    data.max_length=8192 \
    data.batch_size=64 \
    actor.model_name=meta-llama/Llama-3.2-1B-Instruct \
    actor.sp_size=2 \
    actor.max_length_per_device=4096 \
    actor.lr=1e-5 \
    trainer.project=synthetic_gsm \
    trainer.experiment_name=llama-3.2-1b-regenerated-inst \
    trainer.n_epochs=4

# bash examples/synthetic/eval.sh ckpts/llama-3.2-1b-vs-cot-inst
# bash examples/synthetic/eval.sh ckpts/llama-3.2-1b-direct-cot-inst
bash examples/synthetic/eval.sh ckpts/llama-3.2-1b-regenerated-inst