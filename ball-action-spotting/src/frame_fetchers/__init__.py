from src.frame_fetchers.abstract import AbstractFrameFetcher
from src.frame_fetchers.opencv import OpencvFrameFetcher
try:
    from src.frame_fetchers.nvdec import NvDecFrameFetcher
except ImportError:
    print("Using openvcframeetcher")
    NvDecFrameFetcher = None  # VPF not installed — fine, we use OpencvFrameFetcher instead
