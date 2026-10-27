from pathlib import Path
from typing import Any

import torch
import PyNvVideoCodec as nvc

from src.frame_fetchers.abstract import AbstractFrameFetcher


class NvDecFrameFetcher(AbstractFrameFetcher):
    def __init__(self, video_path: str | Path, gpu_id: int):
        super().__init__(video_path, gpu_id)

        self._decoder = nvc.SimpleDecoder(
            str(self.video_path),
            gpu_id=gpu_id,
            use_device_memory=True,
            output_color_type=nvc.OutputColorType.RGB,
        )

        metadata = self._decoder.get_stream_metadata()
        self.num_frames = len(self._decoder)
        self.width = metadata.width
        self.height = metadata.height

    def _next_decode(self) -> Any:
        index = self._current_index + 1
        if index >= self.num_frames:
            raise RuntimeError("End of frames")
        return self._decoder[index]

    def _seek_and_decode(self, index: int) -> Any:
        return self._decoder[index]

    def _convert(self, frame: Any) -> torch.Tensor:
        rgb = torch.from_dlpack(frame)

        if rgb.ndim != 3:
            raise ValueError(f"Unexpected frame shape: {tuple(rgb.shape)}")

        if rgb.shape[-1] == 3:
            r, g, b = rgb.unbind(dim=-1)
        elif rgb.shape[0] == 3:
            r, g, b = rgb.unbind(dim=0)
        else:
            raise ValueError(f"Expected RGB frame, got {tuple(rgb.shape)}")

        gray = (
            0.299 * r.float()
            + 0.587 * g.float()
            + 0.114 * b.float()
        ).round().clamp(0, 255).to(torch.uint8)

        return gray
