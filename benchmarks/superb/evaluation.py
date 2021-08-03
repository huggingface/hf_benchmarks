from dataclasses import asdict
from typing import Dict, List

import requests
from datasets import load_dataset, load_metric

from evaluate.schemas import Evaluation, Metric, Task


def evaluate(evaluation_dataset: str, submission_dataset: str, use_auth_token: str) -> List[Dict[str, List]]:
    # Extract task associated with submission dataset
    header = {"Authorization": "Bearer " + use_auth_token}
    response = requests.get(f"http://huggingface.co/api/datasets/{submission_dataset}", headers=header)
    info = response.json()
    task_name = [t.split(":")[1] for t in info["tags"] if t.split(":")[0] == "task"][0]

    evaluation_ds = load_dataset(evaluation_dataset, task_name, split="test", use_auth_token=use_auth_token)
    # TODO(lewtun): Use dataset loading script instead of relying on hard-coded paths
    submission_ds = load_dataset(
        "json",
        data_files=f"https://huggingface.co/datasets/{submission_dataset}/resolve/main/preds.jsonl",
        split="train",
        use_auth_token=use_auth_token,
    )

    evaluation = Evaluation()

    if task_name == "asr":
        task = Task(name=task_name, type="automatic-speech-recognition")
        wer_metric = load_metric("wer")
        value = wer_metric.compute(predictions=submission_ds["text"], references=evaluation_ds["text"])
        task.metrics.append(Metric(name="wer", type="wer", value=value))
        evaluation.results.append({"task": task})

    return asdict(evaluation)
