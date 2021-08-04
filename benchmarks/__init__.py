from .registration import Benchmark, registry


superb = Benchmark(name="superb")
raft = Benchmark(name="raft")
common = Benchmark(name="common")

registry.register_benchmark(superb)
registry.register_benchmark(raft)
registry.register_benchmark(common)
