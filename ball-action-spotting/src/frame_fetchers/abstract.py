import abc
import logging
from pathlib import Path
from typing import Optional, Any

import torch

logger = logging.getLogger(__name__)


class AbstractFrameFetcher(metaclass=abc.ABCMeta):
    def __init__(self, video_path: str | Path, gpu_id: int):
        self.video_path = Path(video_path)
        self.gpu_id = gpu_id
        self.num_frames = -1
        self.width = -1
        self.height = -1

        self._current_index = -1

    @property
    def current_index(self) -> int:
        return self._current_index

    def fetch_frame(self, index: Optional[int] = None) -> torch.Tensor:
        try:
            if index is None:
                if self._current_index < self.num_frames - 1:
                    frame = self._next_decode()
                    self._current_index += 1
                else:
                    raise RuntimeError("End of frames")
            else:
                if index < 0 or index >= self.num_frames:
                    raise RuntimeError(f"Frame index {index} out of range")
                frame = self._seek_and_decode(index)
                self._current_index = index

            frame = self._convert(frame)
        except BaseException as error:
            logger.error(
                f"Error while fetching frame {index} from '{str(self.video_path)}': {error}."
                f"Replace by empty frame."
            )
            frame = torch.zeros(self.height, self.width,
                                dtype=torch.uint8,
                                device=f"cuda:{self.gpu_id}")
        return frame

    def fetch_frames(self, indexes: list[int]) -> torch.Tensor:
        if not indexes:
            return torch.empty(
                (0, self.height, self.width),
                dtype=torch.uint8,
                device=f"cuda:{self.gpu_id}",
            )

        if any(i < 0 or i >= self.num_frames for i in indexes):
            raise RuntimeError(f"Frame indexes out of range: {indexes}")

        # Preserve the caller's order, including duplicate indexes.
        frames = [self.fetch_frame(i) for i in indexes]
        return torch.stack(frames, dim=0)


    @abc.abstractmethod
    def _next_decode(self) -> Any:
        pass

    @abc.abstractmethod
    def _seek_and_decode(self, index: int) -> Any:
        pass

    @abc.abstractmethod
    def _convert(self, frame: Any) -> torch.Tensor:
        pass
