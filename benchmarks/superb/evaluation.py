from typing import Dict, List

from datasets import load_dataset, load_metric


def evaluate(evaluation_dataset: str, submission_dataset: str, use_auth_token: str, **kwargs) -> List[Dict[str, List]]:
    eval_ds = load_dataset(evaluation_dataset, "asr", split="test").select(range(10))
    sub_ds = load_dataset(
        "json",
        data_files=f"https://huggingface.co/datasets/{submission_dataset}/resolve/main/asr-preds.jsonl",
        split="train",
    )
    wer_metric = load_metric("wer")
    metric = wer_metric.compute(predictions=sub_ds["text"], references=eval_ds["text"])

    metrics = [{"asr": [{"metrics": [{"name": "wer", "value": metric, "split": "test"}]}]}]
    return metrics
