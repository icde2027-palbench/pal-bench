__all__ = ["BenchmarkDatasetBuilder"]


def __getattr__(name: str):
    if name == "BenchmarkDatasetBuilder":
        from .dataset_builder import BenchmarkDatasetBuilder

        return BenchmarkDatasetBuilder
    raise AttributeError(name)
