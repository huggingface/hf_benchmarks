import os
import re
import subprocess
from pathlib import Path

import pandas as pd
import requests
import typer
from dotenv import load_dotenv


if Path(".env").is_file():
    load_dotenv(".env")

auth_token = os.getenv("HF_HUB_TOKEN")
header = {"Authorization": "Bearer " + auth_token}

app = typer.Typer()


def extract_tags(dataset):
    tags = {}
    for tag in dataset["tags"]:
        k, v = tuple(tag.split(":", 1))
        tags[k] = v
    return tags


def get_submission_repos(benchmark: str):
    response = requests.get("http://huggingface.co/api/datasets", headers=header)
    all_datasets = response.json()
    submissions = []

    for dataset in all_datasets:
        tags = extract_tags(dataset)
        # TODO(lewtun): Using prediction-upload might be too restrictive.
        # Using a generic prediction type might be better to cover other benchmarks
        if tags.get("benchmark") == benchmark and tags.get("type") == "prediction-upload":
            submissions.append(dataset)

    return submissions


@app.command()
def run(benchmark: str, evaluation_dataset: str):
    submissions = get_submission_repos(benchmark)
    typer.echo(f"Found {len(submissions)} submissions to evaluate on benchmark {benchmark}")

    for submission in submissions:
        submission_dataset = submission["id"]
        response = requests.get(
            f"http://huggingface.co/api/datasets/{submission_dataset}?full=true",
            headers=header,
        )
        data = response.json()
        # Extract submission timestamp and convert to Unix epoch in nanoseconds
        timestamp = pd.to_datetime(data["lastModified"])
        submission_timestamp = int(timestamp.timestamp() * 10 ** 9)
        # Use the Git commit SHA and timestamp to create submission ID
        submission_id = data["sha"] + "-" + str(submission_timestamp)
        process = subprocess.run(
            [
                "autonlp",
                "benchmark",
                "--eval_name",
                f"{benchmark}",
                "--dataset",
                f"{evaluation_dataset}",
                "--submission",
                f"{submission_dataset}",
                "--submission_id",
                f"{submission_id}",
            ],
            stdout=subprocess.PIPE,
        )
        if process.returncode == -1:
            typer.echo(f"Error launching evaluation job for submission {submission_dataset} on {benchmark} benchmark!")
        else:
            try:
                match_job_id = re.search(r"# (\d+)", process.stdout.decode("utf-8"))
                job_id = match_job_id.group(1)
                typer.echo(
                    f"Successfully launched evaluation job #{job_id} for submission {submission_dataset} on {benchmark} benchmark!"
                )
            except Exception as e:
                typer.echo(f"Could not extract AutoNLP job ID due to error: {e}")


if __name__ == "__main__":
    app()
