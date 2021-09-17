import os

import pandas as pd
import typer

from evaluate import get_benchmark_repos


app = typer.Typer()


@app.command()
def run(
    benchmark: str,
    endpoint: str = "datasets",
    repo_type: str = "prediction",
    start_date: str = None,
    end_date: str = None,
    save_path: str = "./data",
):
    if start_date is None or end_date is None:
        default_start_time = pd.Timestamp.now()
        default_end_time = pd.Timestamp.now() - pd.Timedelta(days=7)
        typer.echo(
            f"Submission window not provided, so using past week from {default_start_time.date()} as default window"
        )
        start_date = str(default_start_time.date())
        end_date = str(default_end_time.date())

    submissions = get_benchmark_repos(
        benchmark=benchmark,
        use_auth_token=True,
        endpoint=endpoint,
        repo_type=repo_type,
        start_date=start_date,
        end_date=end_date,
    )
    typer.echo(f"Found {len(submissions)} submissions for evaluation!")
    df = pd.DataFrame(submissions)
    file_path = os.path.join(save_path, f"{benchmark}_submissions_{start_date}_{end_date}.csv")
    df.to_csv(file_path, index=False)
    typer.echo(f"Saved submissions to {os.path.abspath(file_path)}")


if __name__ == "__main__":
    app()
