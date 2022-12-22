from typing import Dict, List, Union

import pandas as pd
import requests
import typer
from huggingface_hub import HfApi, list_datasets


def delete_repos(repository_ids: List[str], auth_token: str, repo_type: str = "dataset") -> None:
    typer.echo(f"Found {len(repository_ids)} repos to delete")
    for repo_id in repository_ids:
        org, name = repo_id.split("/")
        HfApi().delete_repo(token=auth_token, organization=org, name=name, repo_type=repo_type)
        typer.echo(f"Deleted repo: {repo_id}")


def is_time_between(begin_time: str, end_time: str, check_time: str = None) -> bool:
    # Adapted from: https://stackoverflow.com/questions/10048249/how-do-i-determine-if-current-time-is-within-a-specified-range-using-pythons-da
    # If check time is not given, default to current UTC time
    begin_time = pd.to_datetime(begin_time).tz_localize("UTC")
    end_time = pd.to_datetime(end_time).tz_localize("UTC")
    check_time = pd.to_datetime(check_time) or pd.Timestamp.now()
    if begin_time < end_time:
        return check_time >= begin_time and check_time <= end_time
    else:  # crosses midnight
        return check_time >= begin_time or check_time <= end_time


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
    benchmark: str,
    use_auth_token: Union[bool, str, None] = None,
    repo_type: str = "prediction",
    start_date: Union[str, pd.Timestamp] = None,
    end_date: Union[str, pd.Timestamp] = None,
) -> List[Dict]:
    """Gets the metadata associated with benchmark submission and evaluation repositories.

    Args:
        benchmark: The benchmark name.
        auth_token: The authentication token for the Hugging Face Hub
        repo_type: The type of benchmark repository. Can be `prediction`, `model` or `evaluation`.
        start_date: The timestamp for the start of the submission window.
        end_date: The timestamp for the end of the submission window.

    Returns:
        The benchmark repositories' metadata of a given `repo_type`.
    """
    submissions_to_evaluate = []
    submissions = list_datasets(filter=f"benchmark:{benchmark}", full=True, use_auth_token=use_auth_token)

    # Filter for repos that fall within submission window
    if start_date and end_date:
        submissions = [
            submission for submission in submissions if is_time_between(start_date, end_date, submission.lastModified)
        ]

    for submission in submissions:
        # Filter submission templates which have the submission_name="none" default value
        card_data = submission.cardData
        if (
            card_data.get("benchmark") == benchmark
            and card_data.get("submission_name") != "none"
            and card_data.get("type") == repo_type
        ):
            submissions_to_evaluate.append(submission)

    return submissions_to_evaluate


def download_submissions(header):
    response = requests.get("http://huggingface.co/api/datasets", headers=header)
    all_datasets = response.json()
    submissions = []

    for dataset in all_datasets:
        tags = extract_tags(dataset)
        if tags.get("benchmark") == "gem" and tags.get("type") == "evaluation":
            submissions.append(dataset)
    return submissions


def format_submissions(submissions, header):
    all_scores = []
    for idx, submission in enumerate(submissions):
        submission_id = submission["id"]
        response = requests.get(
            f"http://huggingface.co/api/datasets/{submission_id}?full=true",
            headers=header,
        )
        data = response.json()
        card_data = data["cardData"]
        scores = card_data["model-index"][0]
        all_scores.append(scores)
    return all_scores


def get_auth_headers(token: str, prefix: str = "Bearer"):
    return {"Authorization": f"{prefix} {token}"}


def http_post(path: str, token: str, payload=None, domain: str = None, params=None) -> requests.Response:
    """HTTP POST request to the AutoNLP API, raises UnreachableAPIError if the API cannot be reached"""
    try:
        response = requests.post(
            url=domain + path, json=payload, headers=get_auth_headers(token=token), allow_redirects=True, params=params
        )
    except requests.exceptions.ConnectionError:
        print("❌ Failed to reach AutoNLP API, check your internet connection")
    response.raise_for_status()
    return response


def http_get(
    path: str,
    token: str,
    domain: str = None,
) -> requests.Response:
    """HTTP POST request to the AutoNLP API, raises UnreachableAPIError if the API cannot be reached"""
    try:
        response = requests.get(url=domain + path, headers=get_auth_headers(token=token), allow_redirects=True)
    except requests.exceptions.ConnectionError:
        print("❌ Failed to reach AutoNLP API, check your internet connection")
    response.raise_for_status()
    return response
