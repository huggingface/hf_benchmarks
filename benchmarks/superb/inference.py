import json
import os
from pathlib import Path

import requests
import typer
from dotenv import load_dotenv


if Path(".env").is_file():
    load_dotenv(".env")


def main(model_id: str, dataset_name: str, dataset_config: str, dataset_split: str, dataset_column: str):
    auth_token = os.getenv("HF_HUB_TOKEN")
    if auth_token is None:
        raise ValueError("Hugging Face API token not provided! Please set it as an environment variable.")
    else:
        header = {"Authorization": "Bearer " + auth_token}
    data = {
        "dataset_name": dataset_name,
        "dataset_config": dataset_config,
        "dataset_split": dataset_split,
        "dataset_column": dataset_column,
    }
    endpoint = f"https://api-inference.huggingface.co/bulk/run/cpu/{model_id}"
    response = requests.post(url=endpoint, data=json.dumps(data), headers=header)
    if response.status_code == 200:
        res = response.json()
        typer.echo(
            f"Launching inference job with ID {res['jobid']} for model {model_id} on dataset {dataset_name}! Dataset repo created with name {res['bulk_name']}"
        )
    else:
        typer.echo(f"Could not launch inference job! Response status code: {response.status_code}")


if __name__ == "__main__":
    typer.run(main)
