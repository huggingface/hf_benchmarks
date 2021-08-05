import requests
from datasets import load_dataset, load_metric

from evaluate import Evaluation, Metric, Result, Task


def evaluate(evaluation_dataset: str, submission_dataset: str, use_auth_token: str) -> Evaluation:
    # Extract task associated with submission dataset
    header = {"Authorization": "Bearer " + use_auth_token}
    response = requests.get(f"http://huggingface.co/api/datasets/{submission_dataset}", headers=header)
    info = response.json()
    task_name = [t.split(":")[1] for t in info["tags"] if t.split(":")[0] == "task"][0]

    evaluation_ds = load_dataset(evaluation_dataset, task_name, split="test", use_auth_token=use_auth_token)
    # TODO(lewtun): Use dataset loading script instead of relying on hard-coded paths
    submission_ds = load_dataset(
        "json",
        data_files=[f"https://huggingface.co/datasets/{submission_dataset}/resolve/main/preds.jsonl"],
        split="train",
        use_auth_token=use_auth_token,
    )

    evaluation = Evaluation(results=[])

    if task_name == "asr":
        # Define task
        task = Task(name=task_name, type="automatic-speech-recognition", metrics=[])
        # Compute metrics
        wer_metric = load_metric("wer")
        value = wer_metric.compute(predictions=submission_ds["text"], references=evaluation_ds["text"])  # type: ignore
        task["metrics"].append(Metric(name="wer", type="wer", value=value))
        # Collect results
        result = Result(task=task)
        evaluation["results"].append(result)

    return evaluation
