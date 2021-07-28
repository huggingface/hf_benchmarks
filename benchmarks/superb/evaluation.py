from typing import Dict, List

from datasets import load_dataset, load_metric
from huggingface_hub import HfApi


def evaluate(evaluation_dataset: str, submission_dataset: str, use_auth_token: str) -> List[Dict[str, List]]:
    api = HfApi()
    info = api.dataset_info(submission_dataset, token=use_auth_token)
    task = [t.split(":")[1] for t in info.tags if t.split(":")[0] == "task"][0]
    eval_ds = load_dataset(evaluation_dataset, task, split="test", use_auth_token=use_auth_token)
    sub_ds = load_dataset(
        "json",
        data_files=f"https://huggingface.co/datasets/{submission_dataset}/resolve/main/preds.jsonl",
        split="train",
        use_auth_token=use_auth_token,
    )

    metrics = []
    if task == "asr":
        wer_metric = load_metric("wer")
        metric = wer_metric.compute(predictions=sub_ds["text"], references=eval_ds["text"])
        metrics.append({task: [{"metrics": [{"name": wer_metric.name, "value": metric, "split": "test"}]}]})

    return metrics
