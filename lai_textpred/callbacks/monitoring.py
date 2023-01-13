import time
from typing import Any, Dict

import torch

import lightning

from lai_textpred.moving_average import MovingAverage



class GPUMonitoringCallback(lightning.pytorch.callbacks.Callback):
    def __init__(
        self,
        gpu_memory_logname: str = "gpu_stats/max_memory",
        gpu_util_logname: str = "gpu_stats/utilization",
        time_per_batch_logname: str = "time/seconds_per_iter",
    ):

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

        self.gpu_memory_logname = gpu_memory_logname
        self.gpu_util_logname = gpu_util_logname
        self.time_per_batch_logname = time_per_batch_logname

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
        trainer: "lightning.pytorch.Trainer",
        pl_module: "lightning.pytorch.LightningModule",
        batch: Any,
        batch_idx: int,
    ) -> None:
        self._init_gpu_util_trackers(trainer.world_size)

        metrics = {}

        # only calc time after first batch
        if batch_idx:
            curr_time = time.time()
            time_delta = curr_time - self.last_batch_start_time
            avg_time_delta = torch.tensor(
                trainer.strategy.reduce(time_delta), dtype=torch.float
            )
            self.seconds_per_iter10.update(avg_time_delta)
            self.seconds_per_iter100.update(avg_time_delta)
            self.last_batch_start_time = curr_time

            metrics[self.time_per_batch_logname] = avg_time_delta
            metrics[
                f"{self.time_per_batch_logname}{self._average_postfix(10)}"
            ] = self.seconds_per_iter10.compute()
            metrics[
                f"{self.time_per_batch_logname}{self._average_postfix(100)}"
            ] = self.seconds_per_iter100.compute()

        # collect the metrics on the current rank
        device = trainer.strategy.root_device

        max_memory = torch.tensor(
            torch.cuda.max_memory_allocated(), device=device, dtype=torch.float
        ) / (
            1024**3
        )  # in GB
        torch.cuda.reset_max_memory_allocated()

        # gather the metrics from all processes
        max_memory_total_rank = trainer.strategy.all_gather(max_memory)

        if self.running_utilizations_per_batch:
            curr_utils = sum(self.running_utilizations_per_batch) / len(
                self.running_utilizations_per_batch
            )
            curr_utils_total_rank = trainer.strategy.all_gather(curr_utils)
            self._reset_running_utilizations()

            # the metrics are in an N x 1 tensor where N is the total number of processes
            assert curr_utils_total_rank.size(0) == trainer.world_size
        else:
            curr_utils_total_rank = None

        # bookkeeping and compute statistics for each rank
        for i in range(trainer.world_size):
            metrics[f"{self.gpu_memory_logname}_rank{i}"] = max_memory_total_rank[i]
            if curr_utils_total_rank is not None:
                metrics[f"{self.gpu_util_logname}_rank{i}"] = curr_utils_total_rank[i]
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
                    f"{self.gpu_util_logname}_rank{i}{self._average_postfix(10)}"
                ] = self.gpu_utilizations10[i].compute()
            if curr_update_count > 100:
                metrics[
                    f"{self.gpu_util_logname}_rank{i}{self._average_postfix(100)}"
                ] = self.gpu_utilizations100[i].compute()

        pl_module.log_dict(
            metrics, sync_dist=False, on_step=True, on_epoch=False, rank_zero_only=True
        )

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
        self.running_utilizations_per_batch.append(
            torch.tensor(
                torch.cuda.utilization(),
                device=trainer.strategy.root_device,
                dtype=torch.float,
            )
        )

    def on_save_checkpoint(
        self,
        trainer: "lightning.pytorch.Trainer",
        pl_module: "lightning.pytorch.LightningModule",
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
        trainer: "lightning.pytorch.Trainer",
        pl_module: "lightning.pytorch.LightningModule",
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

    @staticmethod
    def _average_postfix(average_window: int) -> str:
        return f"_averaged{average_window}"
