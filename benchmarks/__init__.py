from .registration import Benchmark, registry


raft = Benchmark(name="raft")

registry.register_benchmark(raft)
