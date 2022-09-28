import os
import time
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
        # Format submission name to comply with AutoTrain API
        # _XXX_ for spaces, _DDD_ for double dashes
        # TODO: remove these dirty hacks - should really apply validation at submission time!
        submission_name = tags["submission_name"].replace(" ", "_XXX_")
        submission_name = submission_name.replace("--", "_DDD_")
        # Extract submission timestamp and convert to Unix epoch in nanoseconds
        timestamp = pd.to_datetime(data["lastModified"])
        submission_timestamp = int(timestamp.tz_localize(None).timestamp())
        # Use the user-generated submission name, Git commit SHA and timestamp to create submission ID
        submission_id = submission_name + "__" + data["sha"][:6] + "__" + str(submission_timestamp)
        # Define AutoTrain payload
        project_config = {}
        # Need a dummy dataset to use the dataset loader in AutoTrain
        project_config["dataset_name"] = "autoevaluator/benchmark-dummy-data"
        project_config["dataset_config"] = "autoevaluator--benchmark-dummy-data"
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
                    "provider": "ovh",
                    "instance_type": "p3",
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
        typer.echo("🎨🎨🎨 Project creation 🎨🎨🎨")
        typer.echo(project_json_resp)

        if project_json_resp["created"]:
            data_payload = {
                "split": 4,  # use "auto" split choice in AutoTrain
                "col_mapping": project_config["col_mapping"],
                "load_config": {"max_size_bytes": 0, "shuffle": False},
                "dataset_id": project_config["dataset_name"],
                "dataset_config": project_config["dataset_config"],
                "dataset_split": project_config["dataset_split"],
            }
            data_json_resp = http_post(
                path=f"/projects/{project_json_resp['id']}/data/dataset",
                payload=data_payload,
                token=AUTOTRAIN_TOKEN,
                domain=AUTOTRAIN_BACKEND_API,
            ).json()
            typer.echo("💾💾💾 Dataset creation 💾💾💾")
            typer.echo(data_json_resp)

            # Process data
            data_proc_json_resp = http_post(
                path=f"/projects/{project_json_resp['id']}/data/start_processing",
                token=AUTOTRAIN_TOKEN,
                domain=AUTOTRAIN_BACKEND_API,
            ).json()
            typer.echo(f"🍪 Start data processing response: {data_proc_json_resp}")

            typer.echo("⏳ Waiting for data processing to complete ...")
            is_data_processing_success = False
            while is_data_processing_success is not True:
                project_status = http_get(
                    path=f"/projects/{project_json_resp['id']}",
                    token=AUTOTRAIN_TOKEN,
                    domain=AUTOTRAIN_BACKEND_API,
                ).json()
                # See database.database.enums.ProjectStatus for definitions of `status`
                if project_status["status"] == 3:
                    is_data_processing_success = True
                    print("✅ Data processing complete!")
                time.sleep(10)
                typer.echo("🥱 Dataset not ready, waiting 10 more seconds ...")

            # Approve training job
            train_job_resp = http_post(
                path=f"/projects/{project_json_resp['id']}/start_training",
                token=AUTOTRAIN_TOKEN,
                domain=AUTOTRAIN_BACKEND_API,
            ).json()
            print(f"🏃 Training job approval response: {train_job_resp}")
            print("🔥 Project and dataset preparation completed!")


if __name__ == "__main__":
    app()
