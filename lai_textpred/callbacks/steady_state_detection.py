import operator
import warnings
from collections import defaultdict, deque
from functools import partial
from typing import Any, Optional

import lightning
import torch
from lightning_utilities.core.imports import compare_version

from lai_textpred.steady_state_utils import (
    calc_total_time_per_node,
    chinchilla_metric_samples,
    is_steady_state,
)


class SteadyStateDetection(lightning.pytorch.callbacks.model_summary.ModelSummary):
    def __init__(
        self,
        target_loss: float,
        batch_size: Optional[int] = None,
        num_params: Optional[int] = None,
        rtol: float = 0.015,
        atol: Optional[float] = None,
        steady_state_det_mode: str = "iter_speed",
        average: Optional[float] = None,
        moving_average_window: int = 10,
        stop_on_steady_state: bool = True,
        steady_state_steps_before_stop: int = 10,
        gpu_util_logname: str = "gpu_stats/utilization",
        time_per_batch_logname: str = "time/seconds_per_iter",
    ):
        super().__init__()

        self.target_loss = target_loss
        self.batch_size = batch_size
        self.num_params: Optional[int] = num_params
        self.steady_state_achieved = False
        self.rtol = rtol
        self.atol = atol

        self.gpu_metrics = defaultdict(partial(deque, maxlen=moving_average_window))
        self.iteration_speeds = deque(maxlen=moving_average_window)
        self.average = average
        self.steady_state_det_mode = steady_state_det_mode
        self.stop_on_steady_state = stop_on_steady_state
        self.steady_state_steps_before_stop = steady_state_steps_before_stop
        self.steady_state_stepped = 0
        self.gpu_util_logname = gpu_util_logname
        self.time_per_batch_logname = time_per_batch_logname

        if steady_state_det_mode == "utilization":
            warnings.warn(
                "Cuda utilization as a proxy metric for steady state may not be optimal. "
                "The actual parameters can differ a lot depending on the cluster configuration and backend "
                "parameters. Please consider using `iter_speed` as the steady state detection mode!"
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

    def on_train_batch_start(
        self,
        trainer: lightning.pytorch.Trainer,
        pl_module: lightning.pytorch.LightningModule,
        batch: Any,
        batch_idx: int,
    ) -> None:
        if self.num_params is None:
            model_summary = self._summary(trainer, pl_module)
            self.num_params = model_summary.trainable_parameters

    def on_train_batch_end(
        self,
        trainer: lightning.pytorch.Trainer,
        pl_module: lightning.pytorch.LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ) -> None:

        if self.batch_size is None:
            self.batch_size = lightning.pytorch.utilities.data.extract_batch_size(batch)

        if self.steady_state_achieved:
            self.steady_state_stepped += 1

        metrics = trainer.callback_metrics

        if trainer.is_global_zero and not self.steady_state_achieved:
            for i in range(trainer.world_size):
                metric_name_cuda = (
                    f"{self.gpu_util_logname}_rank{i}"
                    + self._average_postfix(self.average)
                )

                if metric_name_cuda in trainer.callback_metrics:
                    self.gpu_metrics[i].append(
                        trainer.callback_metrics[metric_name_cuda]
                    )

            metric_name_speed = self.time_per_batch_logname + self._average_postfix(
                self.average
            )
            if metric_name_speed in trainer.callback_metrics:
                self.iteration_speeds.append(
                    trainer.callback_metrics[metric_name_speed]
                )

            self.steady_state_achieved = self._steady_state_func()

        should_stop = False
        # only rank 0 can enter this
        if self.steady_state_achieved:
            speed_per_batch_averaged = metrics[
                self.time_per_batch_logname + self._average_postfix(10)
            ]
            # reflect current number of batches
            tpn = calc_total_time_per_node(
                self.num_samples_required
                - trainer.global_step * self.batch_size * trainer.world_size,
                trainer.world_size,
                self.batch_size,
                speed_per_batch_averaged,
            )

            pl_module.log(
                "estimated_total_time", tpn, sync_dist=False, rank_zero_only=True
            )

            if (
                self.stop_on_steady_state
                and self.steady_state_stepped >= self.steady_state_steps_before_stop
            ):
                print(
                    "Stopping training due to steady state achieved! "
                    f"Projected Time for training: {tpn:} hours on "
                    f"{trainer.num_nodes} nodes with a total of "
                    f"{trainer.world_size} parallel training processes! "
                    f"Speed / Batch (bs={self.batch_size}): {speed_per_batch_averaged} seconds. "
                    f"The GPU utilization is {metrics[self.gpu_util_logname + '_rank0' + self._average_postfix(10)]}% "
                    f"on average."
                )
                should_stop = True

        # only rank0 decides as this is the only one that has the metrics
        stop_tensor = torch.tensor(should_stop, device=trainer.strategy.root_device)

        trainer.strategy.broadcast(stop_tensor, src=0)

        if compare_version("lightning", operator.ge, "2.0.0"):
            global_should_stop = trainer.strategy.reduce_boolean_decision(
                should_stop, all=False
            )
        else:
            # backport of reduce_boolean_decision with all=False to lightning < 2.0.0
            decision = torch.tensor(
                int(should_stop), device=trainer.strategy.root_device
            )
            decision = trainer.strategy.reduce(decision, reduce_op="sum")
            global_should_stop = bool(decision > 0)
        trainer.should_stop = global_should_stop

        pl_module.log(
            "steady_state_achieved",
            torch.tensor(int(self.steady_state_achieved), dtype=torch.float),
            sync_dist=False,
            rank_zero_only=True,
        )

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
            return is_steady_state(
                *self.iteration_speeds, rtol=self.rtol, atol=self.atol
            )
        return False

    @staticmethod
    def _average_postfix(average: Optional[float] = None):
        if average is None:
            return ""
        return f"_averaged{average}"
