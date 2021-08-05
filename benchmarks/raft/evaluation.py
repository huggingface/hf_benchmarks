from datasets import load_dataset, load_metric

from evaluate import Evaluation, Metric, Result, Task


def convert_labels_to_ids(example):
    return {"label": 1} if example["answer"] == "Safety" else {"label": 0}


def evaluate(evaluation_dataset: str, submission_dataset: str, use_auth_token: str) -> Evaluation:
    """Computes metrics for a benchmark.

    Args:
        evaluation_dataset (:obj:`str`): Name of private dataset with ground truth labels.
        submission_dataset (:obj:`str`): Name of user submission dataset with model predictions.
        use_auth_token (:obj:`str`): The API token to access your private dataset on the Hugging Face Hub.

    Returns:
        metrics (:obj:`list`): The evaluation metrics.
    """

    # If your benchmark has multiple tasks, define their names here
    tasks = ["TAISafety-binary"]
    # Load metric
    # TODO(lewtun): add more metrics!
    f1 = load_metric("f1")
    # Define container to store metrics
    evaluation = Evaluation(results=[])
    # Iterate over tasks and build up metrics
    for task in tasks:
        task_data = Task(name=task, type="text-classification", metrics=[])
        # Load datasets associated with task
        evaluation_ds = load_dataset(path=evaluation_dataset, name=task, use_auth_token=use_auth_token)
        submission_ds = load_dataset(path=submission_dataset, name=task, use_auth_token=use_auth_token)
        # Sort by title for alignment
        evaluation_ds = evaluation_ds.sort("title")
        submission_ds = submission_ds.sort("title")
        # Create label IDs
        # TODO(lewtun): use ClassLabel type on Dataset side to skip this
        evaluation_ds = evaluation_ds.map(convert_labels_to_ids)
        submission_ds = submission_ds.map(convert_labels_to_ids)
        # Compute metrics and build up list of dictionaries, one per task in your benchmark
        scores = f1.compute(
            predictions=submission_ds["test"]["label"],
            references=evaluation_ds["test"]["label"],
            average="binary",
        )
        for k, v in scores.items():
            task_data["metrics"].append(Metric(name=k, type=k, value=v))
        # Collect results
        result = Result(task=task_data)
        evaluation["results"].append(result)

    return evaluation
