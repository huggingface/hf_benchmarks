import os
import re
import subprocess
from pathlib import Path

import requests
import typer
from dotenv import load_dotenv


if Path(".env").is_file():
    load_dotenv(".env")

app = typer.Typer()


def get_submission_repos(benchmark: str):
    auth_token = os.getenv("HF_HUB_TOKEN")
    header = {"Authorization": "Bearer " + auth_token}
    response = requests.get("http://huggingface.co/api/datasets", headers=header)
    all_datasets = response.json()
    submissions = []

    for dataset in all_datasets:
        is_benchmark = any([t for t in dataset["tags"] if t.split(":")[1] == f"{benchmark}"])
        if is_benchmark:
            submissions.append(dataset)

    return submissions


@app.command()
def run(benchmark: str, evaluation_dataset: str):
    submissions = get_submission_repos(benchmark)
    typer.echo(f"Found {len(submissions)} submissions to evaluate on benchmark {benchmark}")

    for submission in submissions:
        submission_dataset = submission["id"]
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
            ],
            stdout=subprocess.PIPE,
        )
        if process.returncode == -1:
            typer.echo(f"Error launching evaluation job for submission {submission_dataset} on {benchmark} benchmark!")
        else:
            match_job_id = re.search(r"# (\d+)", process.stdout.decode("utf-8"))
            job_id = match_job_id.group(1)
            typer.echo(
                f"Successfully launched evaluation job #{job_id} for submission {submission_dataset} on {benchmark} benchmark!"
            )


if __name__ == "__main__":
    app()
