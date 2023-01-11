import time
from typing import Any, Dict, List

import lightning as L
import torch
import torchmetrics
from lightning import pytorch as pl


def default_callbacks():
    early_stopping = L.pytorch.callbacks.EarlyStopping(
        monitor="train_loss",
        min_delta=0.00,
        verbose=True,
        mode="min",
    )
    checkpoints = L.pytorch.callbacks.ModelCheckpoint(
        save_top_k=3,
        monitor="train_loss",
        mode="min",
    )
    # return [early_stopping, checkpoints, CustomMonitoringCallback()]
    return [checkpoints, CustomMonitoringCallback()]


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


class CustomMonitoringCallback(L.pytorch.callbacks.Callback):
    def __init__(self):
        super().__init__()
        self.last_batch_start_time = None
        self.gpu_utilizations10 = []
        self.gpu_utilizations100 = []
        self.running_utilizations_per_batch = []

        self.seconds_per_iter10 = MovingAverage(
            sliding_window_size=10, sync_on_compute=False
        )
        self.seconds_per_iter100 = MovingAverage(
            sliding_window_size=100, sync_on_compute=False
        )

    def _reset_running_utilizations(self):
        self.running_utilizations_per_batch = []

    def _init_gpu_util_trackers(self, world_size: int):

        if not self.gpu_utilizations10:
            for _ in range(world_size):
                self.gpu_utilizations10.append(
                    MovingAverage(sliding_window_size=10, sync_on_compute=False)
                )
        if not self.gpu_utilizations100:
            for _ in range(world_size):
                self.gpu_utilizations100.append(
                    MovingAverage(sliding_window_size=100, sync_on_compute=False)
                )



    def on_train_batch_start(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        batch: Any,
        batch_idx: int,
    ) -> None:
        self._init_gpu_util_trackers(trainer.world_size)

        metrics = {}

        # only calc time after first batch
        if batch_idx:
            curr_time = time.time()
            time_delta = curr_time - self.last_batch_start_time
            avg_time_delta = torch.tensor(trainer.strategy.reduce(time_delta), dtype=torch.float)
            self.seconds_per_iter10.update(avg_time_delta)
            self.seconds_per_iter100.update(avg_time_delta)
            self.last_batch_start_time = curr_time

            metrics["train/seconds_per_iter"] = avg_time_delta
            metrics[
                "train/seconds_per_iter_averaged10"
            ] = self.seconds_per_iter10.compute()
            metrics[
                "train/seconds_per_iter_averaged100"
            ] = self.seconds_per_iter100.compute()

        # collect the metrics on the current rank
        device = trainer.strategy.root_device

        max_memory = torch.tensor(torch.cuda.max_memory_allocated(), device=device, dtype=torch.float) / (1024**3) # in GB

        torch.cuda.reset_max_memory_allocated()

        # gather the metrics from all processes
        max_memory_total_rank = trainer.strategy.all_gather(max_memory)

        if self.running_utilizations_per_batch:
            curr_utils = sum(self.running_utilizations_per_batch) / len(self.running_utilizations_per_batch)
            curr_utils_total_rank = trainer.strategy.all_gather(curr_utils)
            self._reset_running_utilizations()

            # the metrics are in an N x 1 tensor where N is the total number of processes
            assert curr_utils_total_rank.size(0) == trainer.world_size
        else:
            curr_utils_total_rank = None

        # bookkeeping and compute statistics for each rank
        for i in range(trainer.world_size):
            metrics[
                f"gpu_stats/max_memory_rank{i}"
            ] = max_memory_total_rank[i]
            if curr_utils_total_rank is not None:
                metrics[
                    f"gpu_stats/utilization_rank{i}"
                ] = curr_utils_total_rank[i]
                self.gpu_utilizations10[i].update(curr_utils_total_rank[i])
                self.gpu_utilizations100[i].update(curr_utils_total_rank[i])

            # update counts have to be the same for 10 and 100 metrics
            # check for protected and public because of https://github.com/Lightning-AI/metrics/pull/1370
            curr_update_count = getattr(
                self.gpu_utilizations10[i],
                "_update_count",
                getattr(self.gpu_utilizations10[i], "update_count", 1),
            )
            if curr_update_count > 10:
                metrics[
                    f"gpu_stats/utilization_rank{i}_averaged10"
                ] = self.gpu_utilizations10[i].compute()
            if curr_update_count > 100:
                metrics[
                    f"gpu_stats/utilization_rank{i}_averaged100"
                ] = self.gpu_utilizations100[i].compute()

        pl_module.log_dict(metrics, sync_dist=False, on_step=True, on_epoch=False, rank_zero_only=True)

        trainer.strategy.barrier()
        self._get_current_utilisation(trainer)
        self.last_batch_start_time = time.time()

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self._get_current_utilisation(trainer)

    def on_before_backward(self, trainer, pl_module, loss):
        self._get_current_utilisation(trainer)

    def on_after_backward(self, trainer, pl_module):
        self._get_current_utilisation(trainer)

    def on_before_optimizer_step(self, trainer, pl_module, optimizer, opt_idx=0):
        self._get_current_utilisation(trainer)

    def on_before_zero_grad(self, trainer, pl_module, optimizer):
        self._get_current_utilisation(trainer)

    def _get_current_utilisation(self, trainer):
        self.running_utilizations_per_batch.append(torch.tensor(torch.cuda.utilization(), device=trainer.strategy.root_device, dtype=torch.float))

    def on_save_checkpoint(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        checkpoint: Dict[str, Any],
    ):
        for name_str in ("gpu_utilizations10", "gpu_utilizations100"):
            checkpoint[name_str] = [
                metric.state_dict() for metric in getattr(self, name_str)
            ]

        for name_str in ("seconds_per_iter10", "seconds_per_iter100"):
            checkpoint[name_str] = getattr(self, name_str).state_dict()

    def on_load_checkpoint(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        checkpoint: Dict[str, Any],
    ):
        self._init_gpu_util_trackers(trainer.world_size)
        for name_str in ("gpu_utilizations10", "gpu_utilizations100"):
            for metric, state in zip(
                getattr(self, name_str), checkpoint.pop(name_str, [])
            ):
                metric.load_state_dict(state)

        for name_str in ("seconds_per_iter10", "seconds_per_iter100"):
            getattr(self, name_str).load_state_dict(checkpoint.pop(name_str, {}))
