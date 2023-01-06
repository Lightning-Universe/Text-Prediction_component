import math
from collections import defaultdict, deque
from functools import partial
from typing import Any, Optional

import lightning.pytorch.callbacks.callback

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

procs_per_cloudcompute = {
    "default": 1,
    "cpu-small": 1,
    "cpu-medium": 1,
    "flow-lite": 1,
    "gpu": 1,
    "gpu-rtx": 1,
    "gpu-rtx-multi": 4,
    "gpu-fast": 1,
    "gpu-fast-multi": 4,
    "gpu-multi": 2,
}

def is_steady_state(*metrics, rtol: float = 0.015, atol: Optional[float] = None):
    mean_metric = sum(metrics)/len(metrics)
    min_metric = min(metrics)
    max_metric = max(metrics)
    return _check_tols(max_metric, mean_metric, rtol, atol) and _check_tols(mean_metric, min_metric, rtol, atol)


def _check_atol(val_a, val_b, atol: Optional[float]):
    return (atol is None) or (abs(val_a - val_b) <= atol)

def _check_rtol(val_a, val_b, rtol: float):
    return abs(val_a - val_b) <= (rtol * val_b)

def _check_tols(val_a, val_b, rtol: float, atol: float):
    return _check_atol(val_a, val_b, atol) and _check_rtol(val_a, val_b, rtol)

def chinchilla_metric_samples(final_loss, num_params):
    return ((1.69 + 406.4/num_params**0.34)/final_loss)**(1/0.27)


def calc_compute_cost_from_num_samples(num_samples, cloud_compute, num_cloud_compute, num_procs, batch_size, time_per_batch):
    effective_bs = num_procs * batch_size
    num_batches = math.ceil(num_samples / effective_bs)
    time_per_batch_hours = time_per_batch / 60 / 60 # convert from seconds to hours
    total_time = num_batches * time_per_batch_hours
    cost_per_cloud_compute = cost_per_cloudcompute_hour[cloud_compute] * total_time
    return cost_per_cloud_compute * num_cloud_compute

class CostEstimationCallback(lightning.pytorch.callbacks.model_summary.ModelSummary):
    def __init__(self, target_loss: float, cloud_compute: str, num_nodes: int, batch_size: int = 1, num_params: Optional[int] = None):
        # TODO: Add rtol and atol with sensible defaults
        super().__init__()

        self.target_loss = target_loss
        self.cloud_compute = cloud_compute
        self.num_nodes = num_nodes
        self.batch_size = batch_size
        self.num_params: Optional[int] = num_params
        self.steady_state_achieved = False

        self.gpu_metrics = defaultdict(partial(deque, maxlen=10))

    @property
    def num_samples_required(self) -> int:
        if self.num_params is None:
            raise ValueError('Cannot calculate the number of samples without the number of model parameters!')

        return chinchilla_metric_samples(self.target_loss, self.num_params)

    def on_fit_start(self, trainer: lightning.pytorch.Trainer, pl_module: lightning.pytorch.LightningModule) -> None:
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

        if not self.steady_state_achieved:
            for i in range(trainer.world_size):
                if f"gpu_stats/utilization_rank{i}_averaged10" in metrics:
                    self.gpu_metrics[i].append(metrics[f"gpu_stats/utilization_rank{i}_averaged10"])

            all_steady_state = trainer.strategy.reduce(is_steady_state(*self.gpu_metrics[i], rtol=self.rtol, atol=self.atol), reduce_op='sum') == trainer.world_size
            self.steady_state_achieved = all_steady_state

        if self.steady_state_achieved:
            cost = calc_compute_cost_from_num_samples(self.num_samples_required, self.cloud_compute, trainer.num_nodes, trainer.world_size, self.batch_size, metrics["train/seconds_per_iter_averaged10"])
            pl_module.log('estimated_cost_in_dollars', cost, sync_dist=False, rank_zero_only=True)


