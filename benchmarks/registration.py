from dataclasses import dataclass


@dataclass
class Benchmark:
    name: str


class BenchmarkRegistry:
    """
    Registry for all registered benchmarks.
    """

    def __init__(self):
        self.benchmarks = {}

    def register_benchmark(self, benchmark):
        """
        Register a benchmark.

        Args:
            benchmark: Benchmark to register.
        """
        name = benchmark.name
        if name in self.benchmarks:
            raise ValueError(f"Benchmark with name {name} already registered.")
        self.benchmarks[name] = benchmark

    def get_benchmark(self, name):
        """
        Get a registered benchmark.

        Args:
            name: Name of the benchmark.

        Returns:
            Benchmark with the given name.
        """
        if name not in self.benchmarks:
            raise ValueError("Benchmark with name {} not registered.".format(name))
        return self.benchmarks[name]

    def list_benchmarks(self):
        """
        List all registered benchmarks.

        Returns:
            List of all registered benchmarks.
        """
        return list(self.benchmarks.values())


registry = BenchmarkRegistry()
