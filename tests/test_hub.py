from unittest import TestCase

from evaluate.hub import extract_tags


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
