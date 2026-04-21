from __future__ import annotations

from ..schemas import DatasetConfig
from .minerva import MinervaAdapter
from .omnivideobench import OmniVideoBenchAdapter
from .videomathqa import VideoMathQAAdapter


BENCHMARK_REGISTRY = {
    "videomathqa": VideoMathQAAdapter,
    "minerva": MinervaAdapter,
    "omnivideobench": OmniVideoBenchAdapter,
}


def get_benchmark_adapter(name: str, config: DatasetConfig):
    key = str(name or "").strip().lower()
    if key not in BENCHMARK_REGISTRY:
        raise KeyError("Unknown benchmark: %s" % name)
    return BENCHMARK_REGISTRY[key](config)
