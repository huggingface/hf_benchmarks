from unittest import TestCase

from huggingface_hub import HfFolder

from evaluate import extract_tags, get_benchmark_repos

from .testing_utils import (
    BOGUS_BENCHMARK_NAME,
    DUMMY_BENCHMARK_NAME,
    DUMMY_EVALUATION_ID,
    DUMMY_MODEL_ID,
    DUMMY_PREDICTION_ID,
)


class ExtractTagsTest(TestCase):
    def test_no_tags(self):
        repo_info = {"modelId": "bert-base-uncased"}
        tags = extract_tags(repo_info)
        self.assertDictEqual(tags, {})

    def test_no_keyed_tags(self):
        repo_info = {"modelId": "bert-base-uncased", "tags": ["exbert"]}
        tags = extract_tags(repo_info)
        self.assertDictEqual(tags, {})

    def test_keyed_tags(self):
        repo_info = {"modelId": "bert-base-uncased", "tags": ["benchmark:glue", "dataset:wikipedia"]}
        tags = extract_tags(repo_info)
        self.assertDictEqual(tags, {"benchmark": "glue", "dataset": "wikipedia"})

    def test_keyed_tags_with_multiple_colons(self):
        repo_info = {"modelId": "bert-base-uncased", "tags": ["benchmark:glue:superglue", "dataset:wikipedia"]}
        tags = extract_tags(repo_info)
        self.assertDictEqual(tags, {"benchmark": "glue:superglue", "dataset": "wikipedia"})

    def test_mixed_tags(self):
        repo_info = {"modelId": "bert-base-uncased", "tags": ["exbert", "benchmark:glue", "dataset:wikipedia"]}
        tags = extract_tags(repo_info)
        self.assertDictEqual(tags, {"benchmark": "glue", "dataset": "wikipedia"})


class GetBenchmarkReposTest(TestCase):
    @classmethod
    def setUpClass(cls):
        """
        Share this valid token in all tests below.
        """
        cls._token = HfFolder.get_token()

    def test_no_datasets_repo(self):
        data = get_benchmark_repos(
            benchmark=BOGUS_BENCHMARK_NAME, use_auth_token=True, endpoint="datasets", repo_type="prediction"
        )
        self.assertEqual(len(data), 0)

    def test_no_models_repo(self):
        data = get_benchmark_repos(
            benchmark=BOGUS_BENCHMARK_NAME, use_auth_token=True, endpoint="models", repo_type="prediction"
        )
        self.assertEqual(len(data), 0)

    def test_prediction_repo(self):
        data = get_benchmark_repos(
            benchmark=DUMMY_BENCHMARK_NAME, use_auth_token=True, endpoint="datasets", repo_type="prediction"
        )
        self.assertEqual(len(data), 1)
        self.assertEqual(data[0]["id"], DUMMY_PREDICTION_ID)

    def test_evaluation_repo(self):
        data = get_benchmark_repos(
            benchmark=DUMMY_BENCHMARK_NAME, use_auth_token=True, endpoint="datasets", repo_type="evaluation"
        )
        self.assertEqual(len(data), 1)
        self.assertEqual(data[0]["id"], DUMMY_EVALUATION_ID)

    def test_model_upload_repo(self):
        data = get_benchmark_repos(
            benchmark=DUMMY_BENCHMARK_NAME, use_auth_token=True, endpoint="models", repo_type="model"
        )
        self.assertEqual(len(data), 1)
        self.assertEqual(data[0]["modelId"], DUMMY_MODEL_ID)