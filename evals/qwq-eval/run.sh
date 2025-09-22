MODEL_NAME=$1
BASE_URL=$2

echo $BASE_URL
echo $MODEL_NAME

BASE_MODEL_NAME=$(basename $MODEL_NAME)
mkdir -p output/${BASE_MODEL_NAME}
# python ./generate_api_answers/infer_multithread.py \
#     --input_file "./data/aime24.jsonl" \
#     --output_file "./output/${BASE_MODEL_NAME}/aime24_bz64.jsonl" \
#     --base_url $BASE_URL \
#     --model_name $MODEL_NAME \
#     --n_samples 64

# python ./generate_api_answers/infer_multithread.py \
#     --input_file "./data/aime25.jsonl" \
#     --output_file "./output/${BASE_MODEL_NAME}/aime25_bz64.jsonl" \
#     --base_url $BASE_URL \
#     --model_name $MODEL_NAME \
#     --n_samples 64

python ./generate_api_answers/infer_multithread.py \
    --input_file "./data/livecodebench_v5.jsonl" \
    --output_file "./output/${BASE_MODEL_NAME}/livecodebench_v5_bz1.jsonl" \
    --base_url $BASE_URL \
    --model_name $MODEL_NAME \
    --n_samples 1

# python ./generate_api_answers/infer_multithread.py \
#     --input_file "./data/ifeval.jsonl" \
#     --output_file "./output/${BASE_MODEL_NAME}/ifeval_bz1.jsonl" \
#     --base_url $BASE_URL \
#     --model_name $MODEL_NAME \
#     --n_samples 1

mkdir -p eval_res/${BASE_MODEL_NAME}

# python ./eval/eval.py \
#     --input_path "./output/${BASE_MODEL_NAME}/aime24_bz64.jsonl" \
#     --cache_path "./eval_res/${BASE_MODEL_NAME}/aime24_bz64.jsonl" \
#     --task_name "math_opensource/aime24" \
#     --consensus \
#     > "./eval_res/${BASE_MODEL_NAME}/aime24_bz64_res_result.txt"

# python ./eval/eval.py \
#     --input_path "./output/${BASE_MODEL_NAME}/aime25_bz64.jsonl" \
#     --cache_path "./eval_res/${BASE_MODEL_NAME}/aime25_bz64.jsonl" \
#     --task_name "math_opensource/aime25" \
#     --consensus \
#     > "./eval_res/${BASE_MODEL_NAME}/aime25_bz64_res_result.txt"

python ./data/process_data.py

python  ./eval/eval.py \
    --input_path "./output/${BASE_MODEL_NAME}/livecodebench_v5_bz1.jsonl" \
    --cache_path "./eval_res/${BASE_MODEL_NAME}/livecodebench_v5_bz1.jsonl" \
    --task_name "livecodebench" > "./eval_res/${BASE_MODEL_NAME}/livecodebench_v5_bz1_res_result.txt"

# python  ./eval/eval.py \
#     --input_path "./output/${BASE_MODEL_NAME}/ifeval_bz1.jsonl" \
#     --cache_path "./eval_res/${BASE_MODEL_NAME}/ifeval_bz1.jsonl" \
#     --task_name "ifeval" > "./eval_res/${BASE_MODEL_NAME}/ifeval_bz1_res_result.txt"

python3 ./eval/collect_results.py \
    --base_dir "./eval_res/${BASE_MODEL_NAME}" \
    --model_name $MODEL_NAME \
    --output_path $OUTPUT_DIR/metrics.csv