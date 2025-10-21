#!/usr/bin/env python3
"""
Convert Hendrycks MATH dataset to RL2 format.

This script replicates the data loading from tinker-cookbook's math_env.py:
- Train: ~12,000 problems from EleutherAI/hendrycks_math (with MATH-500 filtered out)
- Test: 500 problems from HuggingFaceH4/MATH-500

Output format (JSONL):
{
  "prompt": "Problem statement. Write your answer in \\boxed{} format.",
  "extra_info": {
    "answer": "ground_truth_answer"
  }
}

Usage:
    python examples/prepare_math_data.py

Outputs:
    - math_train.jsonl (~12k problems)
    - math_test.jsonl (500 problems)
"""
import json
import re
from datasets import load_dataset, concatenate_datasets, get_dataset_config_names


def extract_boxed(text):
    """Extract answer from \\boxed{...} format.

    Matches tinker-cookbook's math_grading.py:extract_boxed()
    """
    # Find all \boxed{...} patterns
    pattern = r'\\boxed\{([^}]+)\}'
    matches = re.findall(pattern, text)

    if not matches:
        raise ValueError("No boxed answer found")

    # Return the last match (in case there are multiple)
    return matches[-1]


def get_hendrycks_math_test():
    """Load MATH-500 test set.

    This is the standard held-out test set for Hendrycks MATH.
    """
    print("Loading MATH-500 test set...")
    test_dataset = load_dataset("HuggingFaceH4/MATH-500", name="default", split="test")
    return test_dataset


def get_hendrycks_math_train():
    """Load Hendrycks MATH train set with MATH-500 filtered out.

    For Hendrycks MATH, the standard is to use both the "train" and "test" splits
    for training. The "test" split here is NOT the same as the MATH-500 test split,
    which is a commonly-held-out subset of 500 of the below 12.5k problems.

    To construct a clean training set, we filter out problems that exist in the
    MATH-500 test set, resulting in ~12,000 train and 500 test problems.

    This matches tinker-cookbook's _get_hendrycks_math_train() exactly.
    """
    # Get test problems to filter out
    test_dataset = get_hendrycks_math_test()
    test_problems = {problem["problem"] for problem in test_dataset}

    print(f"Filtering out {len(test_problems)} test problems from training set...")

    # Load all configs and splits from Hendrycks MATH
    dataset_name = "EleutherAI/hendrycks_math"
    configs = get_dataset_config_names(dataset_name)

    print(f"Loading {len(configs)} configs from {dataset_name}...")
    pieces = []
    for cfg in configs:
        print(f"  Loading config: {cfg}")
        for split in ("train", "test"):
            ds = load_dataset(dataset_name, name=cfg, split=split)
            # Filter out MATH-500 test problems
            ds = ds.filter(lambda ex: ex["problem"] not in test_problems)
            pieces.append(ds)

    # Concatenate all pieces
    full_dataset = concatenate_datasets(pieces)
    print(f"Total training problems after filtering: {len(full_dataset)}")

    return full_dataset


def convert_to_rl2_format(dataset, output_path):
    """Convert HuggingFace dataset to RL2 JSONL format.

    Args:
        dataset: HuggingFace dataset with 'problem' and 'solution' fields
        output_path: Path to output JSONL file
    """
    print(f"\nConverting to RL2 format: {output_path}")

    with open(output_path, 'w') as f:
        total = 0
        skipped = 0

        for row in dataset:
            try:
                # Extract ground truth answer from solution (in boxed format)
                answer = extract_boxed(row["solution"])

                # Create RL2 format
                # Note: The suffix matches MathEnv.question_suffix() from tinker-cookbook
                rl2_row = {
                    "messages": [
                        {
                            "role": "user",
                            "content": row["problem"] + " Write your answer in \\boxed{} format."
                        }
                    ],
                    "extra_info": {
                        "answer": answer
                    }
                }
                f.write(json.dumps(rl2_row) + '\n')
                total += 1

            except ValueError:
                # Skip problems without valid boxed answers
                skipped += 1
                continue

    print(f"  ✓ Wrote {total} problems ({skipped} skipped due to missing boxed answers)")
    return total


def main():
    print("=" * 60)
    print("Preparing Hendrycks MATH dataset for RL2-Tinker training")
    print("=" * 60)

    # Load datasets
    print("\n[1/3] Loading datasets...")
    train_ds = get_hendrycks_math_train().shuffle(seed=0)  # Match tinker-cookbook seed
    test_ds = get_hendrycks_math_test()

    print(f"\nDataset sizes:")
    print(f"  Train: {len(train_ds)} problems")
    print(f"  Test:  {len(test_ds)} problems")

    # Convert to RL2 format
    print("\n[2/3] Converting to RL2 JSONL format...")
    train_count = convert_to_rl2_format(train_ds, "math_train.jsonl")
    test_count = convert_to_rl2_format(test_ds, "math_test.jsonl")

    # Summary
    print("\n[3/3] Summary:")
    print(f"  ✓ math_train.jsonl: {train_count} problems")
    print(f"  ✓ math_test.jsonl:  {test_count} problems")
    print("\nReady for RL2-Tinker training!")
    print("=" * 60)


if __name__ == "__main__":
    main()
