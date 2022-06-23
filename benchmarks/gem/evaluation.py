import json
import subprocess
from typing import List

from huggingface_hub import hf_hub_download  # type: ignore


def compute_metrics(evaluation_dataset: str, submission_dataset: str, use_auth_token: str) -> List[dict]:
    """Computes metrics for a benchmark.

    Args:
        evaluation_dataset (:obj:`str`): Name of private dataset with ground truth labels.
        submission_dataset (:obj:`str`): Name of user submission dataset with model predictions.
        use_auth_token (:obj:`str`): The API token to access your private dataset on the Hugging Face Hub.

    Returns:
        metrics (:obj:`List[dict]`): The evaluation metrics.
    """
    # AutoTrain runs the evaluation job inside a Docker container, so we need to
    # save the metrics in the root directory to avoid permission errors.
    metrics_filepath = "/app/metrics.json"
    # This assumes that the GEM submissions are a single file, with a predefined name
    # We'll need to enforce this on the submission repositories
    submission_filename = "submission.json"
    submission_filepath = hf_hub_download(
        repo_id=submission_dataset, filename=submission_filename, repo_type="dataset", use_auth_token=use_auth_token
    )
    # gem_metrics automatically downloads the evaluation splits from the Hub
    process = subprocess.run(
        ["gem_metrics", f"{submission_filepath}", "-o", f"{metrics_filepath}"], stdout=subprocess.PIPE
    )
    if process.returncode == -1:
        raise ValueError(f"Error running gem_metrics for submission {submission_dataset} on {evaluation_dataset}!")
    else:
        with open(metrics_filepath, "r") as f:
            metrics = json.load(f)

    return [metrics]
