from typing import Dict, List

from datasets import load_dataset, load_metric


# IMPLEMENT THIS
def evaluate(evaluation_dataset: str, submission_dataset: str, use_auth_token: str) -> List[Dict[str, List]]:
    """Computes metrics for a benchmark.

    Args:
        evaluation_dataset (:obj:`str`): Name of private dataset with ground truth labels.
        submission_dataset (:obj:`str`): Name of user submission dataset with model predictions.
        use_auth_token (:obj:`str`): The API token to access your private dataset on the Hugging Face Hub.

    Returns:
        metrics (:obj:`dict`): The evaluation metrics.
    """

    # If your benchmark has multiple tasks, define their names here
    tasks = ["task1", "task2"]
    # Load metric
    metric = load_metric("your_metric_name")
    # Iterate over tasks and build up metrics
    for task in tasks:
        # Load datasets associated with task
        evaluation_ds = load_dataset(path=evaluation_dataset, name=task, use_auth_token=use_auth_token)
        submission_ds = load_dataset(path=submission_dataset, name=task, use_auth_token=use_auth_token)
        # Compute metrics and build up list of dictionaries, one per task in your benchmark

    metrics = [
        {
            "task1": [
                {"metrics": [{"name": "accuracy", "value": 0.5281704271462356}], "split": "train"},
                {"metrics": [{"name": "accuracy", "value": 0.5988223201408363}], "split": "test"},
            ]
        },
        {
            "task2": [
                {"metrics": [{"name": "accuracy", "value": 0.3076726598083077}], "split": "train"},
                {"metrics": [{"name": "accuracy", "value": 0.6108661656928662}], "split": "test"},
            ]
        },
    ]
    return metrics
