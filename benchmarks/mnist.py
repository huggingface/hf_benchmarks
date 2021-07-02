from datasets import load_dataset, load_metric


def evaluate(test_dataset, submission_dataset, task, **kwargs):
    test_ds = load_dataset(test_dataset)
    preds_ds = load_dataset(submission_dataset, task)
    acc = load_metric("accuracy")
    metrics = {}
    for split in test_ds.keys():
        metrics[split] = acc.compute(
            predictions=preds_ds[split]["preds"], references=test_ds[split]["label"]
        )

    return metrics
