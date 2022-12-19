import time
from typing import Any

import lightning as L
import torch.cuda
import torchmetrics
from lightning import pytorch as pl


def default_callbacks():
    early_stopping = L.pytorch.callbacks.EarlyStopping(
        monitor="val_loss",
        min_delta=0.00,
        verbose=True,
        mode="min",
    )
    checkpoints = L.pytorch.callbacks.ModelCheckpoint(
        save_top_k=3,
        monitor="val_loss",
        mode="min",
    )
    return [early_stopping, checkpoints, CustomMonitoringCallback()]


class MovingAverage(torchmetrics.Metric):
    sliding_window: list[torch.Tensor]
    current_average: torch.Tensor

    def __init__(
        self,
        sliding_window_size: int,
    ) -> None:
        super().__init__()

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


class CustomMonitoringCallback(L.pytorch.callbacks.Callback):
    def __init__(self):
        super().__init__()
        self.last_batch_start_time = None
        self.gpu_utilizations10 = [
            MovingAverage(sliding_window_size=10)
            for _ in range(torch.cuda.device_count())
        ]
        self.gpu_utilizations100 = [
            MovingAverage(sliding_window_size=100)
            for _ in range(torch.cuda.device_count())
        ]

    def on_train_batch_start(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        batch: Any,
        batch_idx: int,
    ) -> None:
        if self.last_batch_start_time is None:
            self.last_batch_start_time = time.time()
            metrics = {}
        else:
            curr_time = time.time()
            time_delta = curr_time - self.last_batch_start_time
            metrics = {"train{separator}seconds per iter": time_delta}
            self.last_batch_start_time = curr_time

        for i in range(torch.cuda.device_count()):
            metrics[
                "gpu_stats{separator}max_memory_gpu" + str(i)
            ] = torch.cuda.max_memory_allocated(device=f"cuda:{i}")
            torch.cuda.reset_peak_memory_stats(f"cuda:{i}")
            curr_utils = torch.cuda.utilization(f"cuda:{i}")
            metrics["gpu_stats{separator}utilization_gpu" + str(i)] = curr_utils
            self.gpu_utilizations10[i].update(curr_utils)
            self.gpu_utilizations100[i].update(curr_utils)

            # update counts have to be the same for 10 and 100 metrics
            # check for protected and public because of https://github.com/Lightning-AI/metrics/pull/1370
            curr_update_count = getattr(
                self.gpu_utilizations10[i],
                "_update_count",
                getattr(self.gpu_utilizations10[i], "update_count", 1),
            )
            if batch_idx % 10 == 0 and curr_update_count > 10:
                metrics[
                    "gpu_stats{separator}utilization_gpu" + str(i) + "_averaged10"
                ] = self.gpu_utilizations10[i].compute()
            if curr_update_count % 100 == 0 and curr_update_count > 100:
                metrics[
                    "gpu_stats{separator}utilization_gpu" + str(i) + "_averaged100"
                ] = self.gpu_utilizations100[i].compute()

        for logger in trainer.loggers:
            separator = logger.group_separator
            logger_metrics = {
                k.format(separator=separator): v for k, v in metrics.items()
            }
            logger.log_metrics(
                metrics=logger_metrics,
                step=trainer.fit_loop.epoch_loop._batches_that_stepped,
            )
