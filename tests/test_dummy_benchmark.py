import importlib
import os
from unittest import TestCase

from huggingface_hub import HfFolder

from .testing_utils import DUMMY_PRIVATE_LABELS_ID, DUMMY_SUBMISSION_ID


class DummyBenchmarkTest(TestCase):
    @classmethod
    def setUpClass(cls):
        """
        Share this valid token in all tests below. Needed for CI
        """
        token = os.getenv("HF_TOKEN")
        if token:
            HfFolder.save_token(token)

    def test_compute_metrics(self):
        eval_module = importlib.import_module("benchmarks.dummy.evaluation")
        token = HfFolder.get_token()
        results = eval_module.compute_metrics(DUMMY_PRIVATE_LABELS_ID, DUMMY_SUBMISSION_ID, use_auth_token=token)
        expected_results = {
            "results": [
                {
                    "task": {
                        "name": "default",
                        "type": "text-classification",
                        "metrics": [{"name": "f1", "type": "f1", "value": 0.5}],
                    }
                }
            ]
        }
        self.assertDictEqual(expected_results, results)
