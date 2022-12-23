import os
from unittest import TestCase

import pandas as pd
from huggingface_hub import HfFolder

from hf_benchmarks import get_benchmark_repos

from .testing_utils import BOGUS_BENCHMARK_NAME, DUMMY_BENCHMARK_NAME, DUMMY_EVALUATION_ID, DUMMY_SUBMISSION_ID


class GetBenchmarkReposTest(TestCase):
    @classmethod
    def setUpClass(cls):
        """
        Share this valid token in all tests below. Needed for CI
        """
        token = os.getenv("HF_TOKEN")
        if token:
            HfFolder.save_token(token)

    def test_no_datasets_repo(self):
        data = get_benchmark_repos(benchmark=BOGUS_BENCHMARK_NAME, use_auth_token=True, repo_type="prediction")
        self.assertEqual(len(data), 0)

    def test_prediction_repo(self):
        data = get_benchmark_repos(benchmark=DUMMY_BENCHMARK_NAME, use_auth_token=True, repo_type="prediction")
        self.assertEqual(len(data), 1)
        self.assertEqual(data[0].id, DUMMY_SUBMISSION_ID)

    def test_evaluation_repo(self):
        data = get_benchmark_repos(benchmark=DUMMY_BENCHMARK_NAME, use_auth_token=True, repo_type="evaluation")
        self.assertEqual(data[0].id, DUMMY_EVALUATION_ID)

    def test_repo_in_submission_window(self):
        repo = get_benchmark_repos(benchmark=DUMMY_BENCHMARK_NAME, use_auth_token=True, repo_type="prediction")
        submission_time = pd.to_datetime(repo[0].lastModified)
        start_date = (submission_time - pd.Timedelta(days=1)).date()
        end_date = (submission_time + pd.Timedelta(days=1)).date()
        data = get_benchmark_repos(
            benchmark=DUMMY_BENCHMARK_NAME,
            use_auth_token=True,
            repo_type="prediction",
            start_date=start_date,
            end_date=end_date,
        )
        self.assertEqual(len(data), 1)
        self.assertEqual(data[0].id, DUMMY_SUBMISSION_ID)

    def test_repo_outside_submission_window(self):
        repo = get_benchmark_repos(benchmark=DUMMY_BENCHMARK_NAME, use_auth_token=True, repo_type="prediction")
        submission_time = pd.to_datetime(repo[0].lastModified)
        start_date = (submission_time + pd.Timedelta(days=1)).date()
        end_date = (submission_time + pd.Timedelta(days=2)).date()
        data = get_benchmark_repos(
            benchmark=DUMMY_BENCHMARK_NAME,
            use_auth_token=True,
            repo_type="prediction",
            start_date=start_date,
            end_date=end_date,
        )
        self.assertEqual(len(data), 0)
