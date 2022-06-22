import json
import subprocess
from typing import List

from huggingface_hub import cached_download, hf_hub_url  # type: ignore


def compute_metrics(evaluation_dataset: str, submission_dataset: str, use_auth_token: str) -> List[dict]:
    """Computes metrics for a benchmark.

    Args:
        evaluation_dataset (:obj:`str`): Name of private dataset with ground truth labels.
        submission_dataset (:obj:`str`): Name of user submission dataset with model predictions.
        use_auth_token (:obj:`str`): The API token to access your private dataset on the Hugging Face Hub.

    Returns:
        metrics (:obj:`List[dict]`): The evaluation metrics.
    """
    metrics_filename = "metrics.json"
    # This assumes that the GEM submissions are a single file, with a predefined name
    # We'll need to enforce this on the submission repositories
    submission_filename = "submission.json"

    submission_url = hf_hub_url(submission_dataset, submission_filename, repo_type="dataset")
    submission_filepath = cached_download(submission_url, use_auth_token=use_auth_token)
    # gem_metrics automatically downloads the evaluation splits from the Hub
    process = subprocess.run(
        ["sudo", "gem_metrics", f"{submission_filepath}", "-o", f"{metrics_filename}"], stdout=subprocess.PIPE
    )
    if process.returncode == -1:
        raise ValueError(f"Error running gem_metrics for submission {submission_dataset} on {evaluation_dataset}!")
    else:
        with open(metrics_filename, "r") as f:
            metrics = json.load(f)

    return [metrics]
