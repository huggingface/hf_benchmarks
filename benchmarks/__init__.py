from .registration import Benchmark, registry


raft = Benchmark(name="raft")
gem = Benchmark(name="gem")
dummy = Benchmark(name="dummy")

registry.register_benchmark(raft)
registry.register_benchmark(gem)
registry.register_benchmark(dummy)
