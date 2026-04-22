from .extractors import ObservationExtractor

__all__ = ["ObservationExtractor", "ToolRegistry"]


def __getattr__(name: str):
    if name == "ToolRegistry":
        from .registry import ToolRegistry

        return ToolRegistry
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
