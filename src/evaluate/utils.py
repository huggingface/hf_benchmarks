from datasets import import_main_class, prepare_module


def get_benchmark_tasks(dataset):
    module, _ = prepare_module(dataset)
    builder_cls = import_main_class(module)
    return list(builder_cls.builder_configs.keys())
