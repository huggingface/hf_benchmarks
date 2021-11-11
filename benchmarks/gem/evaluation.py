import json
import subprocess

from huggingface_hub import cached_download, hf_hub_url

from evaluate import Evaluation


def compute_metrics(evaluation_dataset: str, submission_dataset: str, use_auth_token: str) -> Evaluation:
    """Computes metrics for a benchmark.

    Args:
        evaluation_dataset (:obj:`str`): Name of private dataset with ground truth labels.
        submission_dataset (:obj:`str`): Name of user submission dataset with model predictions.
        use_auth_token (:obj:`str`): The API token to access your private dataset on the Hugging Face Hub.

    Returns:
        evaluation (:obj:`Evaluation`): The evaluation metrics.
    """
    metrics_filename = "metrics.json"

    references_url = hf_hub_url(evaluation_dataset, "references.json", repo_type="dataset")
    predictions_url = hf_hub_url(submission_dataset, "predictions.json", repo_type="dataset")
    references_filepath = cached_download(references_url)
    predictions_filepath = cached_download(predictions_url)

    process = subprocess.run(
        ["gem_metrics", "-r", f"{references_filepath}", f"{predictions_filepath}", "-o", f"{metrics_filename}"],
        stdout=subprocess.PIPE,
    )
    if process.returncode == -1:
        raise ValueError(f"Error running gem_metrics for submission {submission_dataset} on {evaluation_dataset}!")
    else:
        with open(metrics_filename, "r") as f:
            metrics = json.load(f)

    return metrics
