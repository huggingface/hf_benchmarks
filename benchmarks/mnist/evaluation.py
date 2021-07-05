from collections import defaultdict
from typing import Dict

from datasets import load_dataset, load_metric
import json


def evaluate(
    test_dataset: str = "mnist",
    submission_dataset: str = "lewtun/mnist-preds",
    **kwargs
) -> Dict[str, Dict[str, Dict[str, float]]]:
    metrics = defaultdict(dict)
    tasks = ["task1", "task2"]
    test_ds = load_dataset(test_dataset)
    for task in tasks:
        preds_ds = load_dataset(submission_dataset, task)
        acc = load_metric("accuracy")
        for split in test_ds.keys():
            metrics[task][split] = acc.compute(
                predictions=preds_ds[split]["preds"], references=test_ds[split]["label"]
            )
    return metrics


def main():
    metrics = evaluate()

    with open("metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f)


if __name__ == "__main__":
    main()
