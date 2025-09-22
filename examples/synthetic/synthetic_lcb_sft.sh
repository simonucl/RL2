MODELS=(
    meta-llama/Llama-3.1-8B-Instruct
    # Qwen/Qwen3-4B-Base
    # Qwen/Qwen2.5-7B
    # Qwen/Qwen2.5-7B-Instruct
)
DATASETS=(
    # "simonycl/gsm8k_training_positive_1k_transformed"
    # "simonycl/gsm8k_training_positive_direct_1k_transformed"
    # "simonycl/gsm8k_training_positive_direct_cot_1k_transformed"
    # "simonycl/gsm8k_training_positive_vs_standard_1k_transformed"
    # "simonycl/gsm8k_training_positive_vs_cot_transformed"
    # "simonycl/gsm8k_training_positive_sequence_1k_transformed"
    # "simonycl/gsm8k_training_positive_direct_multi_turn_1k_transformed"
    # "simonycl/gsm8k_training_positive_vs_cot_transformed"
    # "simonycl/gsm8k_training_vs_multi_1k_transformed"
    # meta-llama/Llama-3.1-8B-Instruct
    simonycl/lcb_training_synthetic_positive_direct
    simonycl/lcb_training_positive_sequence
    simonycl/lcb_training_positive_vs_standard
    simonycl/lcb_training_positive_vs_cot
)
# for model in ${MODELS[@]}; do
#     for dataset in ${DATASETS[@]}; do
#         experiment_name=$(echo ${dataset} | sed 's/simonycl\///g')
#         model_name=$(echo ${model} | sed 's/\//-/g')

#         echo saving to ckpts/${model_name}_${experiment_name}
#         torchrun \
#             --nproc_per_node=4 \
#             -m RL2.trainer.sft \
#             data.path=${dataset} \
#             data.max_length=8192 \
#             data.batch_size=64 \
#             actor.model_name=${model} \
#             actor.sp_size=2 \
#             actor.max_length_per_device=4096 \
#             actor.lr=1e-5 \
#             trainer.project=synthetic_lcb \
#             trainer.experiment_name=${experiment_name} \
#             trainer.n_epochs=3 \
#             trainer.save_dir=ckpts/${model_name}_${experiment_name}

#         sleep 15 # for cleaning up the cache
#     done
# done

for model in ${MODELS[@]}; do
    for dataset in ${DATASETS[@]}; do
        experiment_name=$(echo ${dataset} | sed 's/simonycl\///g')
        model_name=$(echo ${model} | sed 's/\//-/g')
        echo evaluating ckpts/${model_name}_${experiment_name}
        bash examples/synthetic/eval_lcb.sh ckpts/${model_name}_${experiment_name}
        sleep 10
    done
done

# bash examples/synthetic/eval_lcb.sh Qwen/Qwen3-4B-Instruct-2507