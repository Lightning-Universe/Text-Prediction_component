from typing import Any, List

import torch
import torchmetrics


class MovingAverage(torchmetrics.Metric):
    # TODO: implement this with collections.deque once other iterables are allowed as state.
    sliding_window: List[torch.Tensor]
    current_average: torch.Tensor

    def __init__(self, sliding_window_size: int, **kwargs) -> None:
        super().__init__(**kwargs)

        # need to add states here globally independent of arguments for momentum and sliding_window_size to satisfy mypy
        self.add_state("sliding_window", [], persistent=True)
        self.sliding_window_size = sliding_window_size

    def update(self, value: torch.Tensor) -> None:

        self.sliding_window.append(value.detach())

        if len(self.sliding_window) > self.sliding_window_size:
            self.sliding_window.pop(0)

    def compute(self) -> torch.Tensor:

        result = sum(self.sliding_window) / len(self.sliding_window)
        if not isinstance(result, torch.Tensor):
            result = torch.tensor(result, device=self.device, dtype=torch.float)
        return result

    def get_extra_state(self) -> Any:
        return {"sliding_window_size": self.sliding_window_size}

    def set_extra_state(self, state: Any) -> None:
        self.sliding_window_size = state.pop("sliding_window_size")
