import argparse
import os
import uuid
from pathlib import Path

import evaluate
import pandas as pd
import yaml
from datasets import get_dataset_config_names, load_dataset
from dotenv import load_dotenv
from huggingface_hub import HfApi
from loguru import logger

from hf_benchmarks import Evaluation, Metric, Result, Task, get_benchmark_repos


MARKDOWN = """---
benchmark: {benchmark}
type: evaluation
submission_id: {submission_id}
submission_dataset: {submission_dataset}
evaluation_dataset: {dataset}
tags:
- autotrain
- benchmark
model-index:
{model_index}
---
# AutoTrain Benchmark

Benchmark: {benchmark}

Evaluation Dataset: {dataset}

Submission Dataset: {submission_dataset}

"""

if Path(".env").is_file():
    load_dotenv(".env")

HF_READ_TOKEN = os.getenv("HF_READ_TOKEN")
HF_WRITE_TOKEN = os.getenv("HF_WRITE_TOKEN")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--end_date", type=str, required=True)
    parser.add_argument("--previous_days", type=int, required=True)
    args = parser.parse_args()

    return args


def compute_metrics(evaluation_dataset: str, submission_dataset: str, token: str) -> Evaluation:
    """Computes metrics for a benchmark.

    Args:
        evaluation_dataset (:obj:`str`): Name of private dataset with ground truth labels.
        submission_dataset (:obj:`str`): Name of user submission dataset with model predictions.
        token (:obj:`str`): The API token to access your private dataset on the Hugging Face Hub.

    Returns:
        evaluation (:obj:`Evaluation`): The evaluation metrics.
    """

    # We need to use the public dataset to get the task names
    tasks = get_dataset_config_names("ought/raft")
    # Load metric
    f1 = evaluate.load("f1")
    # Define container to store metrics
    evaluation = Evaluation(results=[])
    # Iterate over tasks and build up metrics
    for task in sorted(tasks):
        task_data = Task(name=task, type="text-classification", metrics=[])
        # Load datasets associated with task
        evaluation_ds = load_dataset(path=evaluation_dataset, name=task, token=token, split="test")
        submission_ds = load_dataset(path=submission_dataset, name=task, token=token, split="test")
        # Sort IDs to ensure we compare the correct examples
        evaluation_ds = evaluation_ds.sort("ID")
        submission_ds = submission_ds.sort("ID")
        # Compute metrics and build up list of dictionaries, one per task in the benchmark
        scores = f1.compute(
            predictions=submission_ds["Label"],
            references=evaluation_ds["Label"],
            average="macro",
        )
        for k, v in scores.items():
            task_data["metrics"].append(Metric(name=k, type=k, value=v))
        # Collect results
        result = Result(task=task_data)
        evaluation["results"].append(result)

    return evaluation


def generate_model_card(eval_metrics, args):
    """
    Generate model card for the model
    """
    model_index = yaml.dump(eval_metrics)
    markdown = MARKDOWN.format(
        benchmark="raft",
        dataset=args.eval_dataset,
        submission_id=args.submission_id,
        submission_dataset=args.submission_dataset,
        model_index=model_index,
    )
    with open(os.path.join(args.output_dir, "README.md"), "w", encoding="utf-8") as f:
        f.write(markdown)


def run_benchmark(args):
    api = HfApi(token=HF_WRITE_TOKEN)
    job_id = uuid.uuid4().hex[:6]

    # Create output_dir if it doesn't exist
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    eval_metrics = compute_metrics(
        evaluation_dataset=args.eval_dataset,
        submission_dataset=args.submission_dataset,
        token=HF_READ_TOKEN,
    )
    logger.info("Create prediction repo")
    user = args.submission_id.split("__")[0]
    repo_id = f"benchmarks/raft-{user}-{job_id}"
    repo_url = api.create_repo(
        repo_id=repo_id,
        private=True,
        repo_type="dataset",
        exist_ok=True,
    )
    logger.info(f"Created repo: {repo_url}")

    logger.info("Generating model card")
    generate_model_card(eval_metrics, args)

    api.upload_folder(
        folder_path=args.output_dir,
        repo_id=repo_id,
        repo_type="dataset",
    )
    return repo_url


if __name__ == "__main__":
    parsed_args = parse_args()

    end_date = parsed_args.end_date
    start_date = pd.to_datetime(end_date) - pd.Timedelta(days=parsed_args.previous_days)
    logger.info(f"Evaluating submissions on RAFT from {start_date} to {end_date}")
    submissions = get_benchmark_repos("raft", use_auth_token=HF_READ_TOKEN, start_date=start_date, end_date=end_date)
    logger.info(f"Found {len(submissions)} submissions to evaluate on RAFT: {[s.id for s in submissions]}")
    for submission in submissions:
        submission_dataset = submission.id
        logger.info(f"Evaluating submission {submission_dataset}")
        card_data = submission.cardData
        logger.info(f"Submission card data: {card_data}")
        # Format submission name to comply with AutoTrain API
        # _XXX_ for spaces, _DDD_ for double dashes
        # TODO: remove these dirty hacks - should really apply validation at submission time!
        submission_name = card_data.get("submission_name").replace(" ", "_XXX_")
        submission_name = submission_name.replace("--", "_DDD_")
        # Extract submission timestamp and convert to Unix epoch in nanoseconds
        timestamp = pd.to_datetime(submission.lastModified)
        submission_timestamp = int(timestamp.tz_localize(None).timestamp())
        # Use the user-generated submission name, Git commit SHA and timestamp to create submission ID
        submission_id = submission_name + "__" + submission.sha[:6] + "__" + str(submission_timestamp)
        parsed_args.submission_id = submission_id
        parsed_args.output_dir = "data/raft"
        parsed_args.eval_dataset = "ought/raft-private-labels"
        parsed_args.submission_dataset = submission_dataset

        try:
            generated_repo_id = run_benchmark(parsed_args)
            logger.info(f"{generated_repo_id}")
        except Exception as e:
            logger.error(f"Error running submission {submission_dataset}: {e}")
