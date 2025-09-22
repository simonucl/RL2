import argparse
import glob
import json
import os
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
from tqdm import tqdm

try:
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


def normalize_model_name(path):
    parts = path.split("/")
    for part in parts[::-1]:
        if "checkpoint" in part:
            idx = parts.index(part)
            model_name = parts[idx - 1]
            checkpoint = part
            return f"{model_name}-{checkpoint}"
        if "global_step" in part:
            idx = parts.index(part)
            model_name = parts[idx - 1]
            return f"{model_name}-{part}"
    for part in reversed(parts):
        if any(x in part.lower() for x in ["llama", "qwen", "gpt", "mistral"]):
            return f"{part}-checkpoint-final"
    return "unknown_model"


def get_benchmark_name(path):
    parts = path.split("/")
    return parts[-2] if len(parts) > 1 else "unknown_benchmark"


def process_file(metrics_file):
    try:
        model_name = normalize_model_name(metrics_file)
        benchmark = get_benchmark_name(metrics_file)
        with open(metrics_file, "r") as f:
            metrics = json.load(f)
            acc = metrics.get("acc", 0)
        return model_name, benchmark, {"acc": acc}
    except Exception as e:
        print(f"Error processing {metrics_file}: {e}")
        return None


def collect_results(base_dir, num_threads=8):
    results = defaultdict(lambda: defaultdict(dict))
    metrics_files = glob.glob(f"{base_dir}/**/test_*metrics.json", recursive=True)
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = list(
            tqdm(
                executor.map(process_file, metrics_files),
                total=len(metrics_files),
                desc="Processing files",
            )
        )
        for result in futures:
            if result is not None:
                model_name, benchmark, metrics = result
                results[model_name][benchmark] = metrics
    return results


def sync_to_wandb(args, df, csv_path):
    metrics_data = []
    for _, row in df.iterrows():
        key = row.get("key")
        value = row.get("value")
        if pd.isna(key) or pd.isna(value):
            continue
        metrics_data.append({"key": key, "value": value, "step": args.step})
    metrics_df = pd.DataFrame(metrics_data)
    metrics_df.to_csv(csv_path, index=False)


def main(args):
    base_dir = args.base_dir
    benchmarks = set(args.benchmarks.split(",")) if args.benchmarks else None
    print("Collecting results...")
    results = collect_results(base_dir, args.num_threads)
    if benchmarks:
        filtered_results = defaultdict(lambda: defaultdict(dict))
        for model, model_results in results.items():
            for benchmark, metrics in model_results.items():
                if benchmark in benchmarks:
                    filtered_results[model][benchmark] = metrics
        results = filtered_results
    rows = []
    for model, benchmarks_dict in results.items():
        for benchmark, metrics in benchmarks_dict.items():
            row = {"model": model, "benchmark": benchmark}
            row.update(metrics)
            rows.append(row)
    df = pd.DataFrame(rows)
    output_file = args.output_path or os.path.join(base_dir, "collected_results.csv")
    df.to_csv(output_file, index=False)
    print(f"\nResults saved to {output_file}")
    if args.use_wandb:
        print("\nSyncing to wandb...")
        if args.wandb_api_key and WANDB_AVAILABLE:
            import wandb

            wandb.login(key=args.wandb_api_key)
        sync_to_wandb(args, df, output_file)
        print("Wandb sync completed!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir", type=str, required=True)
    parser.add_argument("--output_path", type=str, default=None)
    args = parser.parse_args()
    main(args)
