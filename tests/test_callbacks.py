import lightning
import pytest

from lit_llms.callbacks import (
    GPUMonitoringCallback,
    SteadyStateDetection,
)

from lai_textpred.callbacks import default_callbacks


@pytest.mark.parametrize("target_loss_val", [None, 0.1])
@pytest.mark.parametrize(
    "gpu_util_logname", ["gpu_stats/utilization", "gpu_stats/utilization2"]
)
@pytest.mark.parametrize(
    "time_per_batch_logname", ["time/seconds_per_iter", "time/seconds_per_iter2"]
)
def test_default_callbacks(target_loss_val, gpu_util_logname, time_per_batch_logname):
    kwargs = {
        "target_loss_val": target_loss_val,
        "gpu_util_logname": gpu_util_logname,
        "time_per_batch_logname": time_per_batch_logname,
    }
    assert len(default_callbacks(**kwargs)) == 3 + int(target_loss_val is not None)
    assert isinstance(
        default_callbacks(**kwargs)[0], lightning.pytorch.callbacks.EarlyStopping
    )
    assert isinstance(
        default_callbacks(**kwargs)[1], lightning.pytorch.callbacks.ModelCheckpoint
    )
    assert isinstance(
        default_callbacks(**kwargs)[2], lightning.pytorch.callbacks.Callback
    )
    assert isinstance(default_callbacks(**kwargs)[2], GPUMonitoringCallback)

    if target_loss_val is not None:
        assert isinstance(
            default_callbacks(**kwargs)[3], lightning.pytorch.callbacks.Callback
        )
        assert isinstance(default_callbacks(**kwargs)[3], SteadyStateDetection)

    _, __, *cbs = default_callbacks(**kwargs)
    if target_loss_val is not None:
        assert cbs[0].gpu_util_logname == gpu_util_logname
        assert cbs[0].time_per_batch_logname == time_per_batch_logname
        assert cbs[1].target_loss == target_loss_val
        assert cbs[1].gpu_util_logname == gpu_util_logname
        assert cbs[1].time_per_batch_logname == time_per_batch_logname
    else:
        assert cbs[0].gpu_util_logname == gpu_util_logname
        assert cbs[0].time_per_batch_logname == time_per_batch_logname
