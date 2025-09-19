DATASETS=(
    "simonycl/gsm8k_training_positive_1k_transformed"
    "simonycl/gsm8k_training_positive_direct_1k_transformed"
    "simonycl/gsm8k_training_positive_direct_cot_1k_transformed"
    "simonycl/gsm8k_training_positive_vs_standard_1k_transformed"
    "simonycl/gsm8k_training_positive_vs_cot_transformed"
    "simonycl/gsm8k_training_positive_sequence_1k_transformed"
    "simonycl/gsm8k_training_positive_direct_multi_turn_1k_transformed"
    "simonycl/gsm8k_training_positive_vs_cot_transformed"
    "simonycl/gsm8k_training_vs_multi_1k_transformed"
)
for dataset in ${DATASETS[@]}; do
    experiment_name=$(echo ${dataset} | sed 's/simonycl\///g' | sed 's/_1k_transformed//g' | sed 's/_/ /g' | sed 's/\b\w/\u&/g')
    torchrun \
        --nproc_per_node=4 \
        -m RL2.trainer.sft \
        data.path=${dataset} \
        data.max_length=8192 \
        data.batch_size=64 \
        actor.model_name=meta-llama/Llama-3.1-8B-Instruct \
        actor.sp_size=2 \
        actor.max_length_per_device=4096 \
        actor.lr=1e-5 \
        trainer.project=synthetic_gsm \
        trainer.experiment_name=${experiment_name} \
        trainer.n_epochs=4

    sleep 15 # for cleaning up the cache
done
for dataset in ${DATASETS[@]}; do
    experiment_name=$(echo ${dataset} | sed 's/simonycl\///g' | sed 's/_1k_transformed//g' | sed 's/_/ /g' | sed 's/\b\w/\u&/g')
    bash examples/synthetic/eval.sh ckpts/${experiment_name}
done