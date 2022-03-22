from datasets import get_dataset_config_names, load_dataset, load_metric

from hf_benchmarks import Evaluation, Metric, Result, Task


def compute_metrics(evaluation_dataset: str, submission_dataset: str, use_auth_token: str) -> Evaluation:
    """Computes metrics for a benchmark.

    Args:
        evaluation_dataset (:obj:`str`): Name of private dataset with ground truth labels.
        submission_dataset (:obj:`str`): Name of user submission dataset with model predictions.
        use_auth_token (:obj:`str`): The API token to access your private dataset on the Hugging Face Hub.

    Returns:
        evaluation (:obj:`Evaluation`): The evaluation metrics.
    """

    # We need to use the public dataset to get the task names
    tasks = get_dataset_config_names("ought/raft")
    # Load metric
    f1 = load_metric("f1")
    # Define container to store metrics
    evaluation = Evaluation(results=[])
    # Iterate over tasks and build up metrics
    for task in sorted(tasks):
        task_data = Task(name=task, type="text-classification", metrics=[])
        # Load datasets associated with task
        evaluation_ds = load_dataset(path=evaluation_dataset, name=task, use_auth_token=use_auth_token, split="test")
        submission_ds = load_dataset(path=submission_dataset, name=task, use_auth_token=use_auth_token, split="test")
        # Sort IDs to ensure we compare the correct examples
        evaluation_ds = evaluation_ds.sort("ID")
        submission_ds = submission_ds.sort("ID")
        # Compute metrics and build up list of dictionaries, one per task in the benchmark
        scores = f1.compute(
            predictions=submission_ds["Label"],
            references=evaluation_ds["Label"],
            average="macro",
        )
        for k, v in scores.items():
            task_data["metrics"].append(Metric(name=k, type=k, value=v))
        # Collect results
        result = Result(task=task_data)
        evaluation["results"].append(result)

    return evaluation
