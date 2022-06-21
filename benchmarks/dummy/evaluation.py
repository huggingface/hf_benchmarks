from datasets import load_dataset
from evaluate import load

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

    # Load datasets associated with benchmark
    evaluation_ds = load_dataset(evaluation_dataset, use_auth_token=use_auth_token, split="test")
    submission_ds = load_dataset(submission_dataset, use_auth_token=use_auth_token, split="test")
    # Load metric
    f1 = load("f1")
    # Define container to store metrics
    evaluation = Evaluation(results=[])
    # Compute metrics and build up list of dictionaries, one per task in the benchmark
    task_data = Task(name="default", type="text-classification", metrics=[])
    scores = f1.compute(
        predictions=submission_ds["label"],
        references=evaluation_ds["label"],
        average="macro",
    )
    for k, v in scores.items():
        task_data["metrics"].append(Metric(name=k, type=k, value=v))
    # Collect results
    result = Result(task=task_data)
    evaluation["results"].append(result)

    return evaluation
