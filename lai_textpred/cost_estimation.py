import math
import warnings
from collections import defaultdict, deque
from functools import partial
from typing import Any, Optional

import lightning.pytorch.callbacks.callback
import torch

cost_per_cloudcompute_hour = {
    "default": 0.022,
    "cpu-small": 0.052,
    "cpu-medium": 0.209,
    "flow-lite": 0.01,
    "gpu": 0.132,
    "gpu-rtx": 0.202,
    "gpu-rtx-multi": 0.945,
    "gpu-fast": 0.383,
    "gpu-fast-multi": 1.530,
    "gpu-multi": 0.289,
}


def is_steady_state(*metrics, rtol: float = 0.015, atol: Optional[float] = None):
    mean_metric = sum(metrics) / len(metrics)
    min_metric = min(metrics)
    max_metric = max(metrics)
    return _check_tols(max_metric, mean_metric, rtol, atol) and _check_tols(
        mean_metric, min_metric, rtol, atol
    )


def _check_atol(val_a, val_b, atol: Optional[float]):
    return (atol is None) or (abs(val_a - val_b) <= atol)


def _check_rtol(val_a, val_b, rtol: float):
    return abs(val_a - val_b) <= (rtol * val_b)


def _check_tols(val_a, val_b, rtol: float, atol: float):
    return _check_atol(val_a, val_b, atol) and _check_rtol(val_a, val_b, rtol)


def chinchilla_metric_samples(final_loss, num_params):
    # D = \frac{410N^{0.34}}{N^{0.34}L-1.69N^{0.34}-406.4}^{\frac{1}{0.27}}
    return ((410*num_params**0.34)/(final_loss*num_params**0.34-1.69*num_params**0.34-406.4))**(1/0.27)


def calc_total_time_per_node(num_samples, num_procs, batch_size, time_per_batch):
    num_samples_per_proc = num_samples / num_procs
    num_batches = num_samples_per_proc / batch_size
    return time_per_batch * num_batches / 60 / 60  # time from secs to hours


def calc_compute_cost_from_num_samples(
    num_samples, cloud_compute, num_cloud_compute, num_procs, batch_size, time_per_batch
):
    time_per_node = calc_total_time_per_node(
        num_samples, num_procs, batch_size, time_per_batch
    )
    cost_per_cloud_compute = cost_per_cloudcompute_hour[cloud_compute] * (
        time_per_node)
    return cost_per_cloud_compute * num_cloud_compute


class CostEstimationCallback(lightning.pytorch.callbacks.model_summary.ModelSummary):
    def __init__(
        self,
        target_loss: float,
        cloud_compute: str,
        batch_size: int = 1,
        num_params: Optional[int] = None,
        rtol: float = 0.015,
        atol: Optional[float] = None,
        steady_state_det_mode: str = "iter_speed",
        average: Optional[float] = None,
        moving_average_window: int = 10,
    ):
        # TODO: Add rtol and atol with sensible defaults
        super().__init__()

        self.target_loss = target_loss
        self.cloud_compute = cloud_compute
        self.batch_size = batch_size
        self.num_params: Optional[int] = num_params
        self.steady_state_achieved = False
        self.rtol = rtol
        self.atol = atol

        self.gpu_metrics = defaultdict(partial(deque, maxlen=moving_average_window))
        self.iteration_speeds = deque(maxlen=moving_average_window)
        self.average = average
        self.steady_state_det_mode = steady_state_det_mode

        if steady_state_det_mode == "utilization":
            warnings.warn('Cuda utilization as a proxy metric for steady state may not be optimal. '
                          'The actual parameters can differ a lot depending on the cluster configuration and backend '
                          'parameters. Please consider using `iter_speed` as the steady state detection mode!'
                          )
            self._steady_state_func = self._is_steady_state_utilization
        elif steady_state_det_mode == "iter_speed":
            self._steady_state_func = self._is_steady_state_iteration_speed
        else:
            raise ValueError(
                f"steady_state_det_mode must be either 'utilization' or 'iter_speed', not {steady_state_det_mode}"
            )

    @property
    def num_samples_required(self) -> int:
        if self.num_params is None:
            raise ValueError(
                "Cannot calculate the number of samples without the number of model parameters!"
            )

        return chinchilla_metric_samples(self.target_loss, self.num_params)

    def on_fit_start(
        self,
        trainer: lightning.pytorch.Trainer,
        pl_module: lightning.pytorch.LightningModule,
    ) -> None:
        if self.num_params is None:
            model_summary = self._summary(trainer, pl_module)
            self.num_params = model_summary.trainable_parameters

    def on_train_batch_start(
        self,
        trainer: lightning.pytorch.Trainer,
        pl_module: lightning.pytorch.LightningModule,
        batch: Any,
        batch_idx: int,
    ) -> None:

        metrics = trainer.callback_metrics

        if trainer.is_global_zero and not self.steady_state_achieved:
            for i in range(trainer.world_size):
                metric_name_cuda = (
                    f"gpu_stats/utilization_rank{i}"
                    + self._average_postfix(self.average)
                )

                if metric_name_cuda in trainer.callback_metrics:
                    self.gpu_metrics[i].append(
                        trainer.callback_metrics[metric_name_cuda]
                    )

            metric_name_speed = "train/seconds_per_iter" + self._average_postfix(
                self.average
            )
            if metric_name_speed in trainer.callback_metrics:
                self.iteration_speeds.append(
                    trainer.callback_metrics[metric_name_speed]
                )

            self.steady_state_achieved = self._steady_state_func()

        if self.steady_state_achieved:
            cost = calc_compute_cost_from_num_samples(
                self.num_samples_required,
                self.cloud_compute,
                trainer.num_nodes,
                trainer.world_size,
                self.batch_size,
                metrics["train/seconds_per_iter_averaged10"],
            )
            tpn = calc_total_time_per_node(
                self.num_samples_required,
                trainer.world_size,
                self.batch_size,
                metrics["train/seconds_per_iter_averaged10"],
            )
            pl_module.log(
                "estimated_cost_in_dollars", cost, sync_dist=False, rank_zero_only=True
            )
            pl_module.log(
                "estimated_total_time", tpn, sync_dist=False, rank_zero_only=True
            )
        pl_module.log("steady_state_achieved", torch.tensor(int(self.steady_state_achieved), dtype=torch.float), sync_dist=False, rank_zero_only=True)

    def _is_steady_state_utilization(self):
        steady_states = []
        for i, v in self.gpu_metrics.items():
            if self.gpu_metrics[i].maxlen == len(self.gpu_metrics[i]):
                steady_states.append(
                    is_steady_state(
                        *self.gpu_metrics[i], rtol=self.rtol, atol=self.atol
                    )
                )

        return len(steady_states) == len(self.gpu_metrics) and all(steady_states)

    def _is_steady_state_iteration_speed(
        self,
    ):
        if len(self.iteration_speeds) == self.iteration_speeds.maxlen:
            return is_steady_state(*self.iteration_speeds, rtol=self.rtol, atol=self.atol)
        return False

    @staticmethod
    def _average_postfix(average: Optional[float] = None):
        if average is None:
            return ""
        return f"_averaged{average}"
