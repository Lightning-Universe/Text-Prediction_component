from typing import List, Optional

import lightning

from lit_llms.callbacks.monitoring import GPUMonitoringCallback
from lit_llms.callbacks.steady_state_detection import SteadyStateDetection


def default_callbacks(
    target_loss_val: Optional[float] = None,
    gpu_util_logname: str = "gpu_stats/utilization",
    time_per_batch_logname: str = "time/seconds_per_iter",
) -> List[lightning.pytorch.Callback]:
    early_stopping = lightning.pytorch.callbacks.EarlyStopping(
        monitor="train_loss",
        min_delta=0.00,
        verbose=True,
        mode="min",
    )
    checkpoints = lightning.pytorch.callbacks.ModelCheckpoint(
        save_top_k=3,
        monitor="train_loss",
        mode="min",
    )

    cbs = [
        early_stopping,
        checkpoints,
        GPUMonitoringCallback(
            gpu_util_logname=gpu_util_logname,
            time_per_batch_logname=time_per_batch_logname,
        ),
    ]

    if target_loss_val is not None:
        cbs.append(
            SteadyStateDetection(
                target_loss_val,
                gpu_util_logname=gpu_util_logname,
                time_per_batch_logname=time_per_batch_logname,
            )
        )
    return cbs
