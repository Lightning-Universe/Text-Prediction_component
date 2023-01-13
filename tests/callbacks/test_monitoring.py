import random
from unittest import mock

import lightning
import pytest
import torch

from lai_textpred.callbacks import GPUMonitoringCallback
from lai_textpred.moving_average import MovingAverage
from tests.helpers import setup_ddp


def test_custom_monitoring_callback_init():
    callback = GPUMonitoringCallback()
    assert callback.last_batch_start_time is None
    assert callback.gpu_utilizations10 == []
    assert callback.gpu_utilizations100 == []
    assert callback.running_utilizations_per_batch == []
    assert isinstance(callback.seconds_per_iter10, MovingAverage)
    assert isinstance(callback.seconds_per_iter100, MovingAverage)
    assert callback.seconds_per_iter10.sliding_window_size == 10
    assert callback.seconds_per_iter100.sliding_window_size == 100
    assert not callback.seconds_per_iter10.sync_on_compute
    assert not callback.seconds_per_iter100.sync_on_compute

    callback._init_gpu_util_trackers(15)
    assert len(callback.gpu_utilizations10) == 15
    assert len(callback.gpu_utilizations100) == 15
    for i in range(15):
        assert isinstance(callback.gpu_utilizations10[i], MovingAverage)
        assert isinstance(callback.gpu_utilizations100[i], MovingAverage)
        assert callback.gpu_utilizations10[i].sliding_window_size == 10
        assert callback.gpu_utilizations100[i].sliding_window_size == 100
        assert not callback.gpu_utilizations10[i].sync_on_compute
        assert not callback.gpu_utilizations100[i].sync_on_compute


def _step(callback, trainer, module, batch_idx, world_size):
    callback.on_train_batch_start(trainer, module, None, batch_idx)
    assert callback.last_batch_start_time is not None
    assert len(callback.gpu_utilizations10) == world_size
    assert len(callback.gpu_utilizations100) == world_size
    assert len(callback.running_utilizations_per_batch) == 1
    callback.on_train_batch_end(trainer, module, None, None, batch_idx)
    assert len(callback.running_utilizations_per_batch) == 2
    callback.on_before_backward(trainer, module, None)
    assert len(callback.running_utilizations_per_batch) == 3
    callback.on_after_backward(trainer, module)
    assert len(callback.running_utilizations_per_batch) == 4
    callback.on_before_optimizer_step(trainer, module, None, 0)
    assert len(callback.running_utilizations_per_batch) == 5
    callback.on_before_zero_grad(trainer, module, None)
    assert len(callback.running_utilizations_per_batch) == 6


@mock.patch("torch.cuda.utilization", lambda: random.randint(0, 100))
@mock.patch(
    "torch.cuda.max_memory_allocated", lambda: random.randint(1024**3, 10 * 1024**3)
)
@mock.patch("torch.cuda.reset_max_memory_allocated", lambda: None)
def _custom_monitoring_callback_train_mock(
    rank, world_size, gpu_memory_logname, gpu_util_logname, time_per_batch_logname
):
    setup_ddp(rank, world_size)

    trainer = mock.MagicMock()
    trainer.world_size = world_size
    trainer.global_rank = rank
    strategy = mock.MagicMock()
    strategy.root_device = torch.device("cpu")
    strategy.all_gather = (
        lightning.lite.utilities.distributed._all_gather_ddp_if_available
    )
    trainer.strategy = strategy

    logger = mock.MagicMock()
    logger.group_separator = "/"

    trainer.loggers = [logger]
    module = mock.MagicMock()
    module.log_dict = lambda metrics, **kwargs: logger.log_metrics(metrics=metrics)

    callback = GPUMonitoringCallback(
        gpu_memory_logname=gpu_memory_logname,
        gpu_util_logname=gpu_util_logname,
        time_per_batch_logname=time_per_batch_logname,
    )

    _step(callback, trainer, module, 0, world_size)
    for i in range(world_size):
        assert len(callback.gpu_utilizations10[i].sliding_window) == 0
        assert len(callback.gpu_utilizations100[i].sliding_window) == 0

    assert logger.log_metrics.call_count == 1

    # max memory per rank
    assert len(logger.log_metrics.call_args[-1]["metrics"]) == world_size
    for i in range(world_size):
        assert (
            f"{gpu_memory_logname}_rank{i}"
            in logger.log_metrics.call_args[-1]["metrics"]
        )

    _step(callback, trainer, module, 1, world_size)
    for i in range(world_size):
        assert len(callback.gpu_utilizations10[i].sliding_window) == 1
        assert len(callback.gpu_utilizations100[i].sliding_window) == 1

    assert logger.log_metrics.call_count == 2
    # three times seconds per iter (current, averaged10, averaged100) + utilization per rank + max memory per rank
    assert len(logger.log_metrics.call_args[-1]["metrics"]) == 2 * world_size + 3
    assert time_per_batch_logname in logger.log_metrics.call_args[-1]["metrics"]
    assert (
        f"{time_per_batch_logname}_averaged10"
        in logger.log_metrics.call_args[-1]["metrics"]
    )
    assert (
        f"{time_per_batch_logname}_averaged100"
        in logger.log_metrics.call_args[-1]["metrics"]
    )
    for i in range(world_size):
        assert (
            f"{gpu_util_logname}_rank{i}" in logger.log_metrics.call_args[-1]["metrics"]
        )
        assert (
            f"{gpu_memory_logname}_rank{i}"
            in logger.log_metrics.call_args[-1]["metrics"]
        )

    _step(callback, trainer, module, 2, world_size)
    assert callback.last_batch_start_time is not None
    assert len(callback.gpu_utilizations10) == world_size
    assert len(callback.gpu_utilizations100) == world_size
    for i in range(world_size):
        assert len(callback.gpu_utilizations10[i].sliding_window) == 2
        assert len(callback.gpu_utilizations100[i].sliding_window) == 2

    assert logger.log_metrics.call_count == 3
    # three times seconds per iter (current, averaged10, averaged100) + utilization per rank + max memory per rank
    assert len(logger.log_metrics.call_args[-1]["metrics"]) == 2 * world_size + 3
    assert time_per_batch_logname in logger.log_metrics.call_args[-1]["metrics"]
    assert (
        f"{time_per_batch_logname}_averaged10"
        in logger.log_metrics.call_args[-1]["metrics"]
    )
    assert (
        f"{time_per_batch_logname}_averaged100"
        in logger.log_metrics.call_args[-1]["metrics"]
    )
    for i in range(world_size):
        assert (
            f"{gpu_util_logname}_rank{i}" in logger.log_metrics.call_args[-1]["metrics"]
        )
        assert (
            f"{gpu_memory_logname}_rank{i}"
            in logger.log_metrics.call_args[-1]["metrics"]
        )

    for i in range(10):
        _step(callback, trainer, module, i + 3, world_size)

    assert callback.last_batch_start_time is not None
    assert len(callback.gpu_utilizations10) == world_size
    assert len(callback.gpu_utilizations100) == world_size
    for i in range(world_size):
        assert len(callback.gpu_utilizations10[i].sliding_window) == 10
        assert len(callback.gpu_utilizations100[i].sliding_window) == 12

    assert logger.log_metrics.call_count == 13
    # three times seconds per iter (current, averaged10, averaged100) + utilization per rank + max memory per rank
    # + utilization averaged10 per rank
    assert len(logger.log_metrics.call_args[-1]["metrics"]) == 3 * world_size + 3
    assert time_per_batch_logname in logger.log_metrics.call_args[-1]["metrics"]
    assert (
        f"{time_per_batch_logname}_averaged10"
        in logger.log_metrics.call_args[-1]["metrics"]
    )
    assert (
        f"{time_per_batch_logname}_averaged100"
        in logger.log_metrics.call_args[-1]["metrics"]
    )
    for i in range(world_size):
        assert (
            f"{gpu_util_logname}_rank{i}" in logger.log_metrics.call_args[-1]["metrics"]
        )
        assert (
            f"{gpu_memory_logname}_rank{i}"
            in logger.log_metrics.call_args[-1]["metrics"]
        )
        assert (
            f"{gpu_util_logname}_rank{i}_averaged10"
            in logger.log_metrics.call_args[-1]["metrics"]
        )

    for i in range(100):
        _step(callback, trainer, module, i + 13, world_size)

    assert callback.last_batch_start_time is not None
    assert len(callback.gpu_utilizations10) == world_size
    assert len(callback.gpu_utilizations100) == world_size
    for i in range(world_size):
        assert len(callback.gpu_utilizations10[i].sliding_window) == 10
        assert len(callback.gpu_utilizations100[i].sliding_window) == 100

    assert logger.log_metrics.call_count == 113
    # three times seconds per iter (current, averaged10, averaged100) + utilization per rank + max memory per rank
    # + utilization averaged10 per rank + utilization averaged100 per rank
    assert len(logger.log_metrics.call_args[-1]["metrics"]) == 4 * world_size + 3
    assert time_per_batch_logname in logger.log_metrics.call_args[-1]["metrics"]
    assert (
        f"{time_per_batch_logname}_averaged10"
        in logger.log_metrics.call_args[-1]["metrics"]
    )
    assert (
        f"{time_per_batch_logname}_averaged100"
        in logger.log_metrics.call_args[-1]["metrics"]
    )
    for i in range(world_size):
        assert (
            f"{gpu_util_logname}_rank{i}" in logger.log_metrics.call_args[-1]["metrics"]
        )
        assert (
            f"{gpu_memory_logname}_rank{i}"
            in logger.log_metrics.call_args[-1]["metrics"]
        )
        assert (
            f"{gpu_util_logname}_rank{i}_averaged10"
            in logger.log_metrics.call_args[-1]["metrics"]
        )

        assert (
            f"{gpu_util_logname}_rank{i}_averaged100"
            in logger.log_metrics.call_args[-1]["metrics"]
        )

    print("")


@pytest.mark.parametrize("world_size", [1, 2, 4, 8])
@pytest.mark.parametrize("gpu_util_logname", ["gpu_stats/utilization", "foo/bar_utils"])
@pytest.mark.parametrize("gpu_memory_logname", ["gpu_stats/max_memory", "foo/bar_mem"])
@pytest.mark.parametrize(
    "time_per_batch_logname", ["time/seconds_per_iter", "foo/bar_time"]
)
def test_monitoring_callback_train(
    world_size, gpu_util_logname, gpu_memory_logname, time_per_batch_logname
):
    torch.multiprocessing.spawn(
        _custom_monitoring_callback_train_mock,
        args=(world_size, gpu_memory_logname, gpu_util_logname, time_per_batch_logname),
        nprocs=world_size,
    )


@pytest.mark.parametrize("world_size", [1, 2, 4, 42])
def test_monitoring_checkpoint(world_size):
    trainer = mock.MagicMock()
    trainer.world_size = world_size
    cb = GPUMonitoringCallback()
    cb._init_gpu_util_trackers(trainer.world_size)
    ckpt = {}
    cb.on_save_checkpoint(mock.MagicMock(), mock.MagicMock(), ckpt)
    assert (
        len(ckpt) == 4
    )  # gpu_utilizations10, gpu_utilizations100, seconds_per_iter10, seconds_per_iter100

    assert len(ckpt["gpu_utilizations10"]) == world_size
    assert len(ckpt["gpu_utilizations100"]) == world_size

    cb2 = GPUMonitoringCallback()
    assert not cb2.gpu_utilizations10
    assert not cb2.gpu_utilizations100
    assert not cb2.seconds_per_iter10.sliding_window
    assert not cb2.seconds_per_iter100.sliding_window

    cb2.on_load_checkpoint(trainer, mock.MagicMock(), ckpt)
    assert len(cb2.gpu_utilizations10) == world_size
    assert len(cb2.gpu_utilizations100) == world_size
    assert cb2.seconds_per_iter10.sliding_window_size == 10
    assert cb2.seconds_per_iter100.sliding_window_size == 100

    for i in range(world_size):
        assert len(cb2.gpu_utilizations10[i].sliding_window) == 0
        assert len(cb2.gpu_utilizations100[i].sliding_window) == 0
        assert len(cb2.seconds_per_iter10.sliding_window) == 0
        assert len(cb2.seconds_per_iter100.sliding_window) == 0

    cb.seconds_per_iter10.update(torch.tensor(42.0))
    cb.seconds_per_iter100.update(torch.tensor(42.0))
    for i in range(world_size):
        cb.gpu_utilizations10[i].update(torch.tensor(42.0))
        cb.gpu_utilizations100[i].update(torch.tensor(42.0))

    ckpt2 = {}
    cb.on_save_checkpoint(trainer, mock.MagicMock(), ckpt2)
    assert (
        len(ckpt2) == 4
    )  # gpu_utilizations10, gpu_utilizations100, seconds_per_iter10, seconds_per_iter100

    assert len(ckpt2["gpu_utilizations10"]) == world_size
    assert len(ckpt2["gpu_utilizations100"]) == world_size

    cb2.on_load_checkpoint(trainer, mock.MagicMock(), ckpt2)
    assert len(cb2.gpu_utilizations10) == world_size
    assert len(cb2.gpu_utilizations100) == world_size
    assert cb2.seconds_per_iter10.sliding_window_size == 10
    assert cb2.seconds_per_iter100.sliding_window_size == 100

    for i in range(world_size):
        assert len(cb2.gpu_utilizations10[i].sliding_window) == 1
        assert len(cb2.gpu_utilizations100[i].sliding_window) == 1
        assert len(cb2.seconds_per_iter10.sliding_window) == 1
        assert len(cb2.seconds_per_iter100.sliding_window) == 1
