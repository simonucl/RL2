KL_COEF=0.01
DATASET_MODEL=gpt-4.1
# DATASET_MODEL=gemini-2.5-flash
NEGATIVE_LABEL_SCALE=0.05
# # torchrun \
# #     --nproc_per_node=4 \
# #     -m RL2.trainer.offline_rl \
# #     data.path=simonycl/gsm8k_training_negative_direct_1k_gpt-4.1_transformed \
# #     data.max_length=8192 \
# #     data.batch_size=64 \
# #     data.label_threshold=0.0 \
# #     actor.model_name=meta-llama/Llama-3.2-1B-Instruct \
# #     actor.sp_size=4 \
# #     actor.max_length_per_device=4096 \
# #     actor.lr=1e-5 \
# #     offline_rl.positive_label_scale=1.0 \
# #     trainer.project=synthetic_gsm \
# #     trainer.experiment_name=llama-3.2-1b-positive-only \
# #     trainer.n_epochs=3

# # sleep 30

# torchrun \
#     --nproc_per_node=4 \
#     -m RL2.trainer.offline_rl \
#     data.path=simonycl/gsm8k_training_negative_direct_1k_${DATASET_MODEL}_transformed \
#     data.max_length=8192 \
#     data.batch_size=64 \
#     actor.model_name=meta-llama/Llama-3.2-1B-Instruct \
#     actor.sp_size=4 \
#     actor.max_length_per_device=4096 \
#     actor.lr=1e-5 \
#     offline_rl.positive_label_scale=1.0 \
#     offline_rl.negative_label_scale=${NEGATIVE_LABEL_SCALE} \
#     trainer.project=synthetic_gsm \
#     trainer.experiment_name=llama-3.2-1b-direct-inst-offline-rl \
#     trainer.n_epochs=3

# sleep 30

# torchrun \
#     --nproc_per_node=4 \
#     -m RL2.trainer.offline_rl \
#     data.path=simonycl/gsm8k_training_negative_sequence_1k_${DATASET_MODEL}_transformed \
#     data.max_length=8192 \
#     data.batch_size=64 \
#     actor.model_name=meta-llama/Llama-3.2-1B-Instruct \
#     actor.sp_size=4 \
#     actor.max_length_per_device=4096 \
#     actor.lr=1e-5 \
#     offline_rl.positive_label_scale=1.0 \
#     offline_rl.negative_label_scale=${NEGATIVE_LABEL_SCALE} \
#     trainer.project=synthetic_gsm \
#     trainer.experiment_name=llama-3.2-1b-sequence-inst-offline-rl \
#     trainer.n_epochs=3

# sleep 30

# torchrun \
#     --nproc_per_node=4 \
#     -m RL2.trainer.offline_rl \
#     data.path=simonycl/gsm8k_training_negative_vs_standard_1k_${DATASET_MODEL}_transformed \
#     data.max_length=8192 \
#     data.batch_size=64 \
#     actor.model_name=meta-llama/Llama-3.2-1B-Instruct \
#     actor.sp_size=4 \
#     actor.max_length_per_device=4096 \
#     actor.lr=1e-5 \
#     offline_rl.positive_label_scale=1.0 \
#     offline_rl.negative_label_scale=${NEGATIVE_LABEL_SCALE} \
#     trainer.project=synthetic_gsm \
#     trainer.experiment_name=llama-3.2-1b-vs-standard-inst-offline-rl \
#     trainer.n_epochs=3

# sleep 30

# torchrun \
#     --nproc_per_node=4 \
#     -m RL2.trainer.offline_rl \
#     data.path=simonycl/gsm8k_training_negative_combined_1k_${DATASET_MODEL}_transformed \
#     data.max_length=8192 \
#     data.batch_size=64 \
#     actor.model_name=meta-llama/Llama-3.2-1B-Instruct \
#     actor.sp_size=4 \
#     actor.max_length_per_device=4096 \
#     actor.lr=1e-5 \
#     offline_rl.positive_label_scale=1.0 \
#     offline_rl.negative_label_scale=${NEGATIVE_LABEL_SCALE} \
#     trainer.project=synthetic_gsm \
#     trainer.experiment_name=llama-3.2-1b-combined-inst-offline-rl \
#     trainer.n_epochs=3

MODELS=(
    "ckpts/llama-3.2-1b-positive-only"
    "ckpts/llama-3.2-1b-direct-inst-offline-rl"
    "ckpts/llama-3.2-1b-sequence-inst-offline-rl"
    "ckpts/llama-3.2-1b-vs-standard-inst-offline-rl"
    "ckpts/llama-3.2-1b-combined-inst-offline-rl"
)

for MODEL in "${MODELS[@]}"; do    
    bash examples/synthetic/eval.sh $MODEL
done