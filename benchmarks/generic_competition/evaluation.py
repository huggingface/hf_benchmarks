import pandas as pd
from huggingface_hub import hf_hub_download  # type: ignore
from sklearn import metrics


def compute_metrics(evaluation_dataset: str, submission_dataset: str, use_auth_token: str, **kwargs):
    """Computes metrics for a benchmark.

    Args:
        evaluation_dataset (:obj:`str`): Name of private dataset with ground truth labels.
        submission_dataset (:obj:`str`): Name of user submission dataset with model predictions.
        use_auth_token (:obj:`str`): The API token to access your private dataset on the Hugging Face Hub.

    Returns:
        evaluation (:obj:`Evaluation`): The evaluation metrics.
    """

    user_id = kwargs.get("user_id", None)
    if user_id is None:
        raise ValueError("user_id is required")

    metric = kwargs.get("metric", "accuracy_score")

    eval_fname = hf_hub_download(
        repo_id=evaluation_dataset,
        filename="solution.csv",
        use_auth_token=use_auth_token,
        repo_type="dataset",
    )
    eval_df = pd.read_csv(eval_fname)

    sub_fname = hf_hub_download(
        repo_id=submission_dataset,
        filename=f"submissions/{user_id}.csv",
        use_auth_token=use_auth_token,
        repo_type="dataset",
    )
    sub_df = pd.read_csv(sub_fname)

    # fetch the metric function
    _metric = getattr(metrics, metric)

    public_ids = eval_df[eval_df.split == "public"].id.values
    private_ids = eval_df[eval_df.split == "private"].id.values

    target_cols = [col for col in eval_df.columns if col not in ["id", "split"]]
    public_score = _metric(
        eval_df[eval_df.id.isin(public_ids)][target_cols],
        sub_df[sub_df.id.isin(public_ids)][target_cols],
    )
    private_score = _metric(
        eval_df[eval_df.id.isin(private_ids)][target_cols],
        sub_df[sub_df.id.isin(private_ids)][target_cols],
    )

    evaluation = {
        "public_score": public_score,
        "private_score": private_score,
    }
    return evaluation
