from .registration import Benchmark, registry


raft = Benchmark(name="raft")
gem = Benchmark(name="gem")

registry.register_benchmark(raft)
registry.register_benchmark(gem)
