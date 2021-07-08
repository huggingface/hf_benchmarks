import json
from typing import Dict, List

from datasets import load_dataset, load_metric


def evaluate(
    test_dataset: str = "mnist", submission_dataset: str = "lewtun/mnist-preds", use_auth_token: bool = False, **kwargs
) -> List[Dict[str, Dict]]:
    metrics = []
    tasks = ["task1", "task2"]
    test_ds = load_dataset(test_dataset, use_auth_token=use_auth_token)
    for task in tasks:
        task_data = {task: []}
        preds_ds = load_dataset(submission_dataset, task, use_auth_token=use_auth_token)
        acc = load_metric("accuracy")
        f1 = load_metric("f1")
        for split in test_ds.keys():
            split_data = {}
            scores1 = acc.compute(predictions=preds_ds[split]["preds"], references=test_ds[split]["label"])
            scores2 = f1.compute(
                predictions=preds_ds[split]["preds"],
                references=test_ds[split]["label"],
                average="macro",
            )
            split_data["split"] = split
            split_data["metrics"] = []
            for score in [scores1, scores2]:
                for k, v in score.items():
                    split_data["metrics"].append({"name": k, "value": v.tolist()})
            task_data[task].append(split_data)
        metrics.append(task_data)
    return metrics


def main():
    metrics = evaluate()

    with open("metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f)


if __name__ == "__main__":
    main()
