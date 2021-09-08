from typing import List

import requests
import typer
from huggingface_hub import HfApi


def extract_tags(dataset):
    tags = {}
    for tag in dataset["tags"]:
        k, v = tuple(tag.split(":", 1))
        tags[k] = v
    return tags


def delete_repos(repository_ids: List[str], auth_token: str, repo_type: str = "dataset"):
    typer.echo(f"Found {len(repository_ids)} repos to delete")
    for repo_id in repository_ids:
        org, name = repo_id.split("/")
        HfApi().delete_repo(token=auth_token, organization=org, name=name, repo_type=repo_type)
        typer.echo(f"Deleted repo: {repo_id}")


def get_benchmark_repos(benchmark: str, auth_token: str, repo_type: str = "prediction"):
    header = {"Authorization": "Bearer " + auth_token}
    response = requests.get("http://huggingface.co/api/datasets", headers=header)
    all_datasets = response.json()
    submissions = []

    for dataset in all_datasets:
        tags = extract_tags(dataset)
        # Filter submission templates which have submission_name = "none" by default
        if (
            tags.get("benchmark") == benchmark
            and tags.get("submission_name") != "none"
            and tags.get("type") == repo_type
        ):
            submissions.append(dataset)

    return submissions
