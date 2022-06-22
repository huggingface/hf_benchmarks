import os
import uuid
from pathlib import Path

import pandas as pd
import requests
import typer
from dotenv import load_dotenv

from hf_benchmarks import extract_tags, get_benchmark_repos, http_get, http_post


if Path(".env").is_file():
    load_dotenv(".env")

HF_TOKEN = os.getenv("HF_TOKEN")
AUTOTRAIN_TOKEN = os.getenv("AUTOTRAIN_TOKEN")
AUTOTRAIN_USERNAME = os.getenv("AUTOTRAIN_USERNAME")
AUTOTRAIN_BACKEND_API = os.getenv("AUTOTRAIN_BACKEND_API")

app = typer.Typer()


@app.command()
def run(
    benchmark: str = "dummy",
    evaluation_dataset: str = "lewtun/benchmarks-dummy-private-labels",
    end_date: str = "2022-06-22",
    previous_days: int = 7,
):
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
        # Format submission name to comply with AutoTrain API
        submission_name = tags["submission_name"].replace(" ", "_XXX_")
        # Extract submission timestamp and convert to Unix epoch in nanoseconds
        timestamp = pd.to_datetime(data["lastModified"])
        submission_timestamp = int(timestamp.tz_localize(None).timestamp())
        # Use the user-generated submission name, Git commit SHA and timestamp to create submission ID
        submission_id = submission_name + "__" + str(uuid.uuid4())[:6] + "__" + str(submission_timestamp)
        # Define AutoTrain payload
        project_config = {}
        # Need a dummy dataset to use the dataset loader in AutoTrain
        project_config["dataset_name"] = "lewtun/imdb-dummy"
        project_config["dataset_config"] = "lewtun--imdb-dummy"
        project_config["dataset_split"] = "train"
        project_config["col_mapping"] = {"text": "text", "label": "target"}
        # Specify benchmark parameters
        project_config["dataset"] = evaluation_dataset
        project_config["model"] = benchmark
        project_config["submission_dataset"] = submission_dataset

        # Create project
        payload = {
            "username": AUTOTRAIN_USERNAME,
            "proj_name": submission_id,
            "task": 1,
            "config": {
                "language": "en",
                "max_models": 5,
                "instance": {
                    "provider": "aws",
                    "instance_type": "ml.g4dn.4xlarge",
                    "max_runtime_seconds": 172800,
                    "num_instances": 1,
                    "disk_size_gb": 150,
                },
                "benchmark": {
                    "dataset": project_config["dataset"],
                    "model": project_config["model"],
                    "submission_dataset": project_config["submission_dataset"],
                },
            },
        }
        project_json_resp = http_post(
            path="/projects/create", payload=payload, token=AUTOTRAIN_TOKEN, domain=AUTOTRAIN_BACKEND_API
        ).json()
        typer.echo(f"Project creation: {project_json_resp}")

        # Upload data
        payload = {
            "split": 4,
            "col_mapping": project_config["col_mapping"],
            "load_config": {"max_size_bytes": 0, "shuffle": False},
        }
        data_json_resp = http_post(
            path=f"/projects/{project_json_resp['id']}/data/{project_config['dataset_name']}",
            payload=payload,
            token=AUTOTRAIN_TOKEN,
            domain=AUTOTRAIN_BACKEND_API,
            params={
                "type": "dataset",
                "config_name": project_config["dataset_config"],
                "split_name": project_config["dataset_split"],
            },
        ).json()
        typer.echo(f"Dataset creation: {data_json_resp}")

        # Run training
        train_json_resp = http_get(
            path=f"/projects/{project_json_resp['id']}/data/start_process",
            token=AUTOTRAIN_TOKEN,
            domain=AUTOTRAIN_BACKEND_API,
        ).json()
        typer.echo(f"Training job response: {train_json_resp}")


if __name__ == "__main__":
    app()
