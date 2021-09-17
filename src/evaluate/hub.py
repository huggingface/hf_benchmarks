from typing import Dict, List

import requests
import typer
from huggingface_hub import HfApi


def delete_repos(repository_ids: List[str], auth_token: str, repo_type: str = "dataset") -> None:
    typer.echo(f"Found {len(repository_ids)} repos to delete")
    for repo_id in repository_ids:
        org, name = repo_id.split("/")
        HfApi().delete_repo(token=auth_token, organization=org, name=name, repo_type=repo_type)
        typer.echo(f"Deleted repo: {repo_id}")


def extract_tags(repo_info: Dict) -> Dict:
    """Extracts the tags from a repository's metadata.

    Args:
        repo_info: The repository's metadata.

    Returns:
        The repository's tags.
    """
    tags = {}
    repo_tags = repo_info.get("tags")
    if repo_tags:
        for repo_tag in repo_tags:
            # Restrict splitting to the first ":" in case the value also contains a ":"
            split_tags = repo_tag.split(":", maxsplit=1)
            if len(split_tags) == 2:
                tags[split_tags[0]] = split_tags[1]
    return tags


def get_benchmark_repos(
    benchmark: str, auth_token: str, endpoint: str = "datasets", repo_type: str = "prediction"
) -> List[Dict]:
    """Gets the metadata associated with benchmark submission and evaluation repositories.

    Args:
        benchmark: The benchmark name.
        auth_token: The authentication token for the Hugging Face Hub
        endpoint: The endpoint to query. Can be `datasets` or `models`.
        repo_type: The type of benchmark repository. Can be `prediction`, `model-upload` or `evaluation`.

    Returns:
        The benchmark repositories' metadata of a given `repo_type`.
    """
    header = {"Authorization": "Bearer " + auth_token}
    params = {"full": True} if endpoint == "models" else None
    response = requests.get(f"http://huggingface.co/api/{endpoint}/", headers=header, params=params)
    response.raise_for_status()
    repos = response.json()
    submissions = []

    for repo in repos:
        tags = extract_tags(repo)
        # Filter submission templates which have the submission_name="none" default value
        if (
            tags.get("benchmark") == benchmark
            and tags.get("submission_name") != "none"
            and tags.get("type") == repo_type
        ):
            submissions.append(repo)

    return submissions
