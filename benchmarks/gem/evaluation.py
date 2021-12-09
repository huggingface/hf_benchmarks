import json
import subprocess

from huggingface_hub import cached_download, hf_hub_url, list_repo_files # type: ignore


def compute_metrics(evaluation_dataset: str, submission_dataset: str, use_auth_token: str) -> dict:
    """Computes metrics for a benchmark.

    Args:
        evaluation_dataset (:obj:`str`): Name of private dataset with ground truth labels.
        submission_dataset (:obj:`str`): Name of user submission dataset with model predictions.
        use_auth_token (:obj:`str`): The API token to access your private dataset on the Hugging Face Hub.

    Returns:
        evaluation (:obj:`dict`): The evaluation metrics.
    """
    metrics_filename = "metrics.json"

    predictions_files = list_repo_files(submission_dataset, repo_type="dataset", token=use_auth_token)
    # TODO(lewtun): Add tests to catch exceptions with multiple files
    predictions_filename = [f for f in predictions_files if ".json" in f][0]
    predictions_url = hf_hub_url(submission_dataset, predictions_filename, repo_type="dataset")
    predictions_filepath = cached_download(predictions_url, use_auth_token=use_auth_token)

    process = subprocess.run(
        ["gem_metrics", f"{predictions_filepath}", "-o", f"{metrics_filename}"],
        stdout=subprocess.PIPE,
    )
    if process.returncode == -1:
        raise ValueError(f"Error running gem_metrics for submission {submission_dataset} on {evaluation_dataset}!")
    else:
        with open(metrics_filename, "r") as f:
            metrics = json.load(f)

    # TODO(lewtun): Test YAML conversion is OK in backend / dataset card
    return metrics
