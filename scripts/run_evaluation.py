import os
from pathlib import Path

import pandas as pd
import requests
import typer
from dotenv import load_dotenv

from hf_benchmarks import extract_tags, get_benchmark_repos, http_post


if Path(".env").is_file():
    load_dotenv(".env")

HF_TOKEN = os.getenv("HF_TOKEN")
AUTOTRAIN_USERNAME = os.getenv("AUTOTRAIN_USERNAME")
AUTOTRAIN_TOKEN = os.getenv("AUTOTRAIN_TOKEN")
AUTOTRAIN_BACKEND_API = os.getenv("AUTOTRAIN_BACKEND_API")

app = typer.Typer()


@app.command()
def run(benchmark: str, evaluation_dataset: str, end_date: str, previous_days: int):
    start_date = pd.to_datetime(end_date) - pd.Timedelta(days=previous_days)
    typer.echo(f"Evaluating submissions on benchmark {benchmark} from {start_date} to {end_date}")
    submissions = get_benchmark_repos(benchmark, use_auth_token=HF_TOKEN, start_date=start_date, end_date=end_date)
    typer.echo(f"Found {len(submissions)} submissions to evaluate on benchmark {benchmark}")
    header = {"Authorization": f"Bearer {HF_TOKEN}"}
    for submission in submissions:
        submission_dataset = submission["id"]
        typer.echo(f"Evaluating submission {submission_dataset}")
        response = requests.get(
            f"http://huggingface.co/api/datasets/{submission_dataset}?full=true",
            headers=header,
        )
        data = response.json()
        # Extract submission name from YAML tags
        tags = extract_tags(data)
        # Extract submission timestamp and convert to Unix epoch in nanoseconds
        timestamp = pd.to_datetime(data["lastModified"])
        submission_timestamp = int(timestamp.timestamp() * 10**9)
        # Use the user-generated submission name, Git commit SHA and timestamp to create submission ID
        submission_id = tags["submission_name"] + "__" + data["sha"] + "__" + str(submission_timestamp)

        payload = {
            "username": AUTOTRAIN_USERNAME,
            "dataset": evaluation_dataset,
            "task": 1,
            "model": benchmark,
            "submission_dataset": submission_dataset,
            "submission_id": submission_id,
            "col_mapping": {},
            "split": "test",
            "config": None,
        }
        json_resp = http_post(
            path="/evaluate/create", payload=payload, token=AUTOTRAIN_TOKEN, domain=AUTOTRAIN_BACKEND_API
        ).json()
        print(f"Submitted evaluation with response: {json_resp}")


if __name__ == "__main__":
    app()
