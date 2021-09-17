from typing import Dict, List

import requests
import typer
from huggingface_hub import HfApi


def delete_repos(repository_ids: List[str], auth_token: str, repo_type: str = "dataset"):
    typer.echo(f"Found {len(repository_ids)} repos to delete")
    for repo_id in repository_ids:
        org, name = repo_id.split("/")
        HfApi().delete_repo(token=auth_token, organization=org, name=name, repo_type=repo_type)
        typer.echo(f"Deleted repo: {repo_id}")


def extract_benchmark_tags(repo_info: Dict) -> Dict:
    """Extracts benchmark tags from a repository's metadata.

    Args:
        repo_info: The repository's metadata.

    Returns:
        The benchmark tags.
    """
    benchmark_tags = {}
    repo_tags = repo_info.get("tags")
    if repo_tags:
        for tag in repo_tags:
            if len(tag.split(":", 1)) == 2:
                k, v = tuple(tag.split(":", 1))
                benchmark_tags[k] = v
    return benchmark_tags


def get_benchmark_repos(
    benchmark: str, auth_token: str, endpoint: str = "datasets", submission_type: str = "prediction"
) -> List[Dict]:
    header = {"Authorization": "Bearer " + auth_token}
    params = {"full": True} if endpoint == "models" else None
    response = requests.get(f"http://huggingface.co/api/{endpoint}/", headers=header, params=params)
    response.raise_for_status()
    repos = response.json()
    submissions = []

    for repo in repos:
        tags = extract_benchmark_tags(repo)
        # Filter submission templates which have the submission_name="none" default value
        if (
            tags.get("benchmark") == benchmark
            and tags.get("submission_name") != "none"
            and tags.get("type") == submission_type
        ):
            submissions.append(repo)

    return submissions
