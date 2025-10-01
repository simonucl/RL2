import json
from concurrent.futures import TimeoutError
from statistics import mean

from pebble import ProcessPool
from tqdm import tqdm


def is_multi_choice(answer):
    for c in answer:
        if c not in ["A", "B", "C", "D", "E"]:
            return False
    return True


def init_fn():
    pass


def work(args):
    i, job = args
    from math_opensource_utils.grader import math_equal_process
    from math_opensource_utils.parser import (STRIP_EXCEPTIONS,
                                              choice_answer_clean,
                                              extract_answer,
                                              parse_ground_truth, strip_string)

    assert len(job["gen"]) == 1
    # total_num = len(job["gen"])

    data_name = job["task"].split("/")[1]
    job["gt_cot"], job["gt"] = parse_ground_truth(job, data_name)

    prediction = extract_answer(job["gen"][0], data_name)
    prediction = strip_string(prediction, skip_unit=data_name in STRIP_EXCEPTIONS)

    # cleaning choice results
    if job["gt"] in ["A", "B", "C", "D", "E"] and prediction not in [
        "A",
        "B",
        "C",
        "D",
        "E",
    ]:
        prediction = choice_answer_clean(prediction)
    elif is_multi_choice(job["gt"]) and not is_multi_choice(prediction):
        # remove any non-choice char
        prediction = "".join([c for c in prediction if c in ["A", "B", "C", "D", "E"]])

    params = [(prediction, job["gt"])]
    result = math_equal_process(params[0])

    return i, float(result), prediction


def compute_scores(jobs, cache_path, consensus=False):
    with tqdm(total=len(jobs)) as pbar:
        with ProcessPool(max_workers=20, initializer=init_fn) as pool:
            future = pool.map(work, list(enumerate(jobs)), timeout=10)
            iterator = future.result()
            while True:
                try:
                    i, result, prediction = next(iterator)
                    jobs[i]["accuracy"] = result
                    jobs[i]["extracted_answer"] = prediction
                    jobs[i]["timeout_cnt"] = 0
                    pbar.update(1)
                except StopIteration:
                    break
                except TimeoutError as error:
                    print(error)
                except Exception as error:
                    print(error.traceback)
                    exit()
        for job in jobs:
            if "accuracy" not in job:
                job["accuracy"] = 0
                job["extracted_answer"] = "Timeout"
                job["timeout_cnt"] = 1
    save_cache(jobs, cache_path)
    if not consensus:
        return mean(x["accuracy"] for x in jobs)
    else:
        cache_path = cache_path.replace(".jsonl", "_consensus.jsonl")
        # Group by prompt and take majority vote
        prompt_groups = {}
        for job in jobs:
            prompt = job["prompt"]
            if prompt not in prompt_groups:
                prompt_groups[prompt] = {
                    "prompt": prompt,
                    "answer": job["answer"],
                    "task": job["task"],
                    "extracted_answer": [],
                    "accuracy": [],
                }
            prompt_groups[prompt]["extracted_answer"].append(job["extracted_answer"])
            prompt_groups[prompt]["accuracy"].append(job["accuracy"])
        # Create new jobs list with consolidated prompt groups
        new_jobs = []
        for prompt, group_data in prompt_groups.items():
            # Count occurrences of each answer
            answer_counts = {}
            for answer in group_data["extracted_answer"]:
                if answer not in answer_counts:
                    answer_counts[answer] = 0
                answer_counts[answer] += 1

            # Find the majority answer
            majority_answer = max(answer_counts.items(), key=lambda x: x[1])[0]

            # Check if the majority answer is correct
            correct = False
            for i, answer in enumerate(group_data["extracted_answer"]):
                if answer == majority_answer:
                    correct = group_data["accuracy"][i]
                    break

            # Create a new consolidated job
            new_job = {
                "prompt": group_data["prompt"],
                "answer": group_data["answer"],
                "task": group_data["task"],
                "extracted_answer": group_data["extracted_answer"],
                "consensus_answer": majority_answer,
                "consensus_accuracy": float(correct),
            }
            new_jobs.append(new_job)

        # Save the consolidated jobs
        save_cache(new_jobs, cache_path)
        return mean(x["consensus_accuracy"] for x in new_jobs)


def save_cache(jobs, cache_path):
    with open(cache_path, "w") as g:
        for job in jobs:
            g.write(json.dumps(job, ensure_ascii=False) + "\n")
            g.flush()


if __name__ == "__main__":
    cache_path = "eval_res/aime25_bz64.jsonl"
    jobs = []
    with open(cache_path, "r") as f:
        for line in f:
            jobs.append(json.loads(line))

    cache_path = cache_path.replace(".jsonl", "_consensus.jsonl")
    # Group by prompt and take majority vote
    prompt_groups = {}
    for job in jobs:
        prompt = job["prompt"]
        if prompt not in prompt_groups:
            prompt_groups[prompt] = {
                "prompt": prompt,
                "answer": job["answer"],
                "task": job["task"],
                "extracted_answer": [],
                "accuracy": [],
            }
        prompt_groups[prompt]["extracted_answer"].append(job["extracted_answer"])
        prompt_groups[prompt]["accuracy"].append(job["accuracy"])
    # Create new jobs list with consolidated prompt groups
    new_jobs = []
    for prompt, group_data in prompt_groups.items():
        # Count occurrences of each answer
        answer_counts = {}
        for answer in group_data["extracted_answer"]:
            if answer not in answer_counts:
                answer_counts[answer] = 0
            answer_counts[answer] += 1

        # Find the majority answer
        majority_answer = max(answer_counts.items(), key=lambda x: x[1])[0]

        # Check if the majority answer is correct
        correct = False
        for i, answer in enumerate(group_data["extracted_answer"]):
            if answer == majority_answer:
                correct = group_data["accuracy"][i]
                break

        # Create a new consolidated job
        new_job = {
            "prompt": group_data["prompt"],
            "answer": group_data["answer"],
            "task": group_data["task"],
            "extracted_answer": group_data["extracted_answer"],
            "consensus_answer": majority_answer,
            "consensus_accuracy": float(correct),
        }
        new_jobs.append(new_job)
    save_cache(new_jobs, cache_path)
    print(mean(x["consensus_accuracy"] for x in new_jobs))
