from datasets import load_dataset, load_metric

from evaluate import Evaluation, Metric, Result, Task


# IMPLEMENT THIS
def evaluate(evaluation_dataset: str, submission_dataset: str, use_auth_token: str) -> Evaluation:
    """Computes metrics for a benchmark.

    Args:
        evaluation_dataset (:obj:`str`): Name of private dataset with ground truth labels.
        submission_dataset (:obj:`str`): Name of user submission dataset with model predictions.
        use_auth_token (:obj:`str`): The API token to access your private datasets on the Hugging Face Hub.

    Returns:
        evaluation (:obj:`dict`): The evaluation metrics.
    """

    # If your benchmark has multiple tasks, define their names here
    tasks = ["task1", "task2"]
    # Load one or more metrics
    your_metric = load_metric("your_metric_name")
    # Iterate over tasks and build up metrics
    evaluation = Evaluation(results=[])
    for task_name in tasks:
        # Create task
        task = Task(name=task_name, type="some-task-type", metrics=[])
        # Load datasets associated with task
        evaluation_ds = load_dataset(path=evaluation_dataset, name=task_name, use_auth_token=use_auth_token)
        submission_ds = load_dataset(path=submission_dataset, name=task_name, use_auth_token=use_auth_token)
        # Compute metrics and build up list of dictionaries, one per task in your benchmark
        value = your_metric.compute(
            predictions=submission_ds["preds_column"], references=evaluation_ds["targets_column"]
        )
        metric = Metric(name="your_metric_name", type="your_metric_name", value=value)
        task["metrics"].append(metric)
        # Collect results
        result = Result(task=task)
        evaluation["results"].append(result)

    return evaluation
