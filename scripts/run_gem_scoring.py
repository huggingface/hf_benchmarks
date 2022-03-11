import os
import zipfile
from pathlib import Path

import numpy as np
import requests
import typer
from dotenv import load_dotenv
from huggingface_hub import Repository, cached_download, hf_hub_url

from evaluate import download_submissions, format_submissions, load_json, save_json


if Path(".env").is_file():
    load_dotenv(".env")

auth_token = os.getenv("HF_HUB_TOKEN")
header = {"Authorization": "Bearer " + auth_token}

SCORES_REPO_URL = "https://huggingface.co/datasets/GEM-submissions/submission-scores"
OUTPUTS_REPO_URL = "https://huggingface.co/datasets/GEM-submissions/outputs-and-scores"
LOCAL_SCORES_REPO = "data/submission-scores"
LOCAL_OUTPUTS_REPO = "data/outputs-and-scores"
GEM_V1_PATH = "data/gem-v1-outputs-and-scores"
# This file is used to configure the filtering of the raw submissions and also used to configure the GEM website
EVAL_CONFIG_URL = (
    "https://raw.githubusercontent.com/GEM-benchmark/GEM-benchmark.github.io/main/web/results/eval_config.json"
)

app = typer.Typer()


def extract_relevant_metrics(config: dict):
    """Extract the `measures` field from the config."""
    metric_names = []
    for k, v in config["measures"].items():
        metric_names.extend(v)
    return metric_names


def drop_unnecessary_metrics(submission_scores: dict, list_of_metrics: list):
    """Return submission_scores with every metric not in list_of_metrics removed."""
    for data_name, data in submission_scores.items():
        if data_name in ["param_count", "submission_name"]:
            continue
        filtered_scores = {k: v for k, v in data.items() if k in list_of_metrics}
        submission_scores[data_name] = filtered_scores
    return submission_scores


def _round_subelements(v):
    """traverses object and rounds items."""
    if isinstance(v, float):
        return round(v, 3)
    elif isinstance(v, int) or isinstance(v, str):
        return v
    elif isinstance(v, dict):
        return {k: (round(d, 3) if isinstance(d, float) else d) for k, d in v.items()}
    else:
        raise ValueError(f"unexpected type: {type(v)}: {v}.")


def round_results(submission_scores: dict):
    """rounds every metric result to three decimal places."""
    for data_name, data in submission_scores.items():
        if data_name in ["param_count", "submission_name"]:
            continue
        rounded_scores = {k: _round_subelements(v) for k, v in data.items()}
        submission_scores[data_name] = rounded_scores
    return submission_scores


def filter_submission_output(submission_scores: dict, config: dict):
    relevant_metrics = extract_relevant_metrics(config)
    filtered_scores = [drop_unnecessary_metrics(d, relevant_metrics) for d in submission_scores]
    return [round_results(d) for d in filtered_scores]


@app.command()
def run():
    # Download submission metadata from the Hub
    hub_submissions = download_submissions(header)
    # Filter out the test submissions
    hub_submissions = [sub for sub in hub_submissions if "lewtun" not in sub["id"]]
    # Download the submission from v1 of the GEM benchmark
    gem_v1_url = hf_hub_url(
        "GEM-submissions/v1-outputs-and-scores", filename="gem-v1-outputs-and-scores.zip", repo_type="dataset"
    )
    gem_v1_path = cached_download(gem_v1_url)
    # Load the submissions from v1
    with zipfile.ZipFile(gem_v1_path) as zf:
        zf.extractall("data")

    gem_v1_files = [p for p in Path(GEM_V1_PATH).glob("*.scores.json")]
    gem_v1_submissions = [load_json(p) for p in gem_v1_files]
    typer.echo(f"Number of submissions from version 1 of the benchmark: {len(gem_v1_submissions)}")
    # Some fields have NaNs which breaks the frontend - replace with -999 as a workaround
    gem_v1_scores = []
    for scores in gem_v1_submissions:
        for k, v in scores.items():
            if isinstance(v, dict):
                for kk, vv in v.items():
                    if "msttr" in kk and np.isnan(vv):
                        scores[k][kk] = -999
        gem_v1_scores.append(scores)
    # Download scores from Hub and combine with v1 scores
    all_scores = format_submissions(hub_submissions, header)
    all_scores.extend(gem_v1_scores)
    typer.echo(f"Number of raw scores: {len(all_scores)}")
    # Clone the Hub repo with the scores
    repo = Repository(
        local_dir=LOCAL_SCORES_REPO,
        clone_from=SCORES_REPO_URL,
        repo_type="dataset",
        use_auth_token=auth_token,
    )
    # Filter the scores for smaller payload to the website / Spaces
    eval_config = requests.get(EVAL_CONFIG_URL).json()
    filtered_scores = filter_submission_output(all_scores, eval_config)
    typer.echo(f"Number of filtered scores: {len(filtered_scores)}")
    if len(all_scores) != len(filtered_scores):
        raise ValueError("The raw and filtered scores must have the same count!")
    # Save and update the raw and filtered scores
    save_json(f"{LOCAL_SCORES_REPO}/scores.json", all_scores)
    save_json(f"{LOCAL_SCORES_REPO}/filtered_scores.json", filtered_scores)

    if repo.is_repo_clean():
        typer.echo("No new submissions were found! Skipping update to the scores repo ...")
    else:
        repo.git_add()
        repo.push_to_hub("Update submission scores")


if __name__ == "__main__":
    app()
