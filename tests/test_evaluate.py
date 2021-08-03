import importlib
import inspect

from benchmarks import registry

# TODO(lewtun): use common.evaluation.evaluate as reference?
EVALUATE_ARGS = {"evaluation_dataset", "submission_dataset", "use_auth_token"}


def test_evaluate_signature():
    benchmarks = registry.list_benchmarks()
    for benchmark in benchmarks:
        evaluate_module = importlib.import_module(f"benchmarks.{benchmark.name}.evaluation")
        args = inspect.signature(evaluate_module.evaluate).parameters.keys()
        assert len(args) == len(EVALUATE_ARGS) and sorted(args) == sorted(EVALUATE_ARGS)
