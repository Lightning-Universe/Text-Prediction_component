import random
from unittest import mock

import lightning
import pytest
import torch

from lai_textpred.callbacks import (
    CustomMonitoringCallback,
    MovingAverage,
    default_callbacks,
)
from tests.helpers import setup_ddp

try:
    from lightning.lite.utilities.distributed import _all_gather_ddp_if_available
except ImportError:
    from lightning.fabric.utilities.distributed import _all_gather_ddp_if_available


def test_default_callbacks():
    assert isinstance(default_callbacks()[0], lightning.pytorch.callbacks.EarlyStopping)
    assert isinstance(
        default_callbacks()[1], lightning.pytorch.callbacks.ModelCheckpoint
    )
    assert isinstance(default_callbacks()[2], lightning.pytorch.callbacks.Callback)
    assert isinstance(default_callbacks()[2], CustomMonitoringCallback)


def test_moving_average():
    ma = MovingAverage(sliding_window_size=5)
    assert ma.sliding_window_size == 5
    assert ma.sliding_window == []

    # not yet updated -> division by length of sliding window is division by zero
    with pytest.raises(ZeroDivisionError):
        ma.compute()

    # sequentially updating
    ma.update(torch.tensor(1.0))
    assert ma.compute() == 1.0
    assert len(ma.sliding_window) == 1
    ma.update(torch.tensor(2.0))
    assert ma.compute() == 1.5
    assert len(ma.sliding_window) == 2
    ma.update(torch.tensor(3.0))
    assert ma.compute() == 2.0
    assert len(ma.sliding_window) == 3

    # resetting -> nothing in sliding window again
    ma.reset()
    assert len(ma.sliding_window) == 0

    # updating again to previous state
    ma.update(torch.tensor(1.0))
    ma.update(torch.tensor(2.0))
    ma.update(torch.tensor(3.0))

    # continue sequentially updating
    ma.update(torch.tensor(4.0))
    assert ma.compute() == 2.5
    assert len(ma.sliding_window) == 4

    ma.update(torch.tensor(5.0))
    assert ma.compute() == 3.0
    assert len(ma.sliding_window) == 5

    # since we are at maximum length,
    # the first item here is popped when a new one is added
    # -> 1.0 is popped -> (2+3+4+5+6)/5 = 20/5 = 4.0
    ma.update(torch.tensor(6.0))
    assert ma.compute() == 4.0
    assert len(ma.sliding_window) == 5

    # sequentially updating (always pops first item)
    ma.update(torch.tensor(7.0))
    assert ma.compute() == 5.0
    assert len(ma.sliding_window) == 5

    ma.update(torch.tensor(8.0))
    assert ma.compute() == 6.0
    assert len(ma.sliding_window) == 5

    ma.update(torch.tensor(9.0))
    assert ma.compute() == 7.0
    assert len(ma.sliding_window) == 5

    ma.update(torch.tensor(10.0))
    assert ma.compute() == 8.0
    assert len(ma.sliding_window) == 5


def test_moving_average_checkpoint():
    ma = MovingAverage(42)

    ma.update(torch.tensor(5.0))
    ma.update(torch.tensor(6.0))
    ma.update(torch.tensor(7.0))
    ma.update(torch.tensor(8.0))
    ma.update(torch.tensor(9.0))
    ma.update(torch.tensor(10.0))

    state_dict = ma.state_dict()
    assert state_dict["sliding_window"] == [
        torch.tensor(5.0),
        torch.tensor(6.0),
        torch.tensor(7.0),
        torch.tensor(8.0),
        torch.tensor(9.0),
        torch.tensor(10.0),
    ]
    assert state_dict["_extra_state"]["sliding_window_size"] == 42

    ma2 = MovingAverage(5)
    ma2.load_state_dict(state_dict)

    assert ma2.sliding_window_size == 42
    assert len(ma2.sliding_window) == 6
    ma2.update(torch.tensor(11.0))


def test_custom_monitoring_callback_init():
    callback = CustomMonitoringCallback()
    assert callback.last_batch_start_time is None
    assert callback.gpu_utilizations10 == []
    assert callback.gpu_utilizations100 == []
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


@mock.patch("torch.cuda.utilization", lambda: random.randint(0, 100))
@mock.patch("torch.cuda.max_memory_allocated", lambda: random.randint(0, 10))
@mock.patch("torch.cuda.reset_max_memory_allocated", lambda: None)
def _custom_monitoring_callback_train_mock(rank, world_size):
    setup_ddp(rank, world_size)

    trainer = mock.MagicMock()
    trainer.world_size = world_size
    trainer.global_rank = rank
    strategy = mock.MagicMock()
    strategy.root_device = torch.device("cpu")
    strategy.all_gather = _all_gather_ddp_if_available
    trainer.strategy = strategy

    logger = mock.MagicMock()
    logger.group_separator = "/"

    trainer.loggers = [logger]
    module = mock.MagicMock()

    callback = CustomMonitoringCallback()

    callback.on_train_batch_start(trainer, module, None, 0)
    assert callback.last_batch_start_time is not None
    assert len(callback.gpu_utilizations10) == world_size
    assert len(callback.gpu_utilizations100) == world_size
    for i in range(world_size):
        assert len(callback.gpu_utilizations10[i].sliding_window) == 1
        assert len(callback.gpu_utilizations100[i].sliding_window) == 1

    assert logger.log_metrics.call_count == 1
    # utilization per rank + max memory per rank
    assert len(logger.log_metrics.call_args[-1]["metrics"]) == 2 * world_size
    for i in range(world_size):
        assert (
            f"gpu_stats/utilization_rank{i}"
            in logger.log_metrics.call_args[-1]["metrics"]
        )
        assert (
            f"gpu_stats/max_memory_rank{i}"
            in logger.log_metrics.call_args[-1]["metrics"]
        )

    callback.on_train_batch_start(trainer, module, None, 1)
    assert callback.last_batch_start_time is not None
    assert len(callback.gpu_utilizations10) == world_size
    assert len(callback.gpu_utilizations100) == world_size
    for i in range(world_size):
        assert len(callback.gpu_utilizations10[i].sliding_window) == 2
        assert len(callback.gpu_utilizations100[i].sliding_window) == 2

    assert logger.log_metrics.call_count == 2
    # three times seconds per iter (current, averaged10, averaged100) + utilization per rank + max memory per rank
    assert len(logger.log_metrics.call_args[-1]["metrics"]) == 2 * world_size + 3
    assert "train/seconds_per_iter" in logger.log_metrics.call_args[-1]["metrics"]
    assert (
        "train/seconds_per_iter_averaged10"
        in logger.log_metrics.call_args[-1]["metrics"]
    )
    assert (
        "train/seconds_per_iter_averaged100"
        in logger.log_metrics.call_args[-1]["metrics"]
    )
    for i in range(world_size):
        assert (
            f"gpu_stats/utilization_rank{i}"
            in logger.log_metrics.call_args[-1]["metrics"]
        )
        assert (
            f"gpu_stats/max_memory_rank{i}"
            in logger.log_metrics.call_args[-1]["metrics"]
        )

    for i in range(10):
        callback.on_train_batch_start(trainer, module, None, i + 3)

    assert callback.last_batch_start_time is not None
    assert len(callback.gpu_utilizations10) == world_size
    assert len(callback.gpu_utilizations100) == world_size
    for i in range(world_size):
        assert len(callback.gpu_utilizations10[i].sliding_window) == 10
        assert len(callback.gpu_utilizations100[i].sliding_window) == 12

    assert logger.log_metrics.call_count == 12
    # three times seconds per iter (current, averaged10, averaged100) + utilization per rank + max memory per rank
    # + utilization averaged10 per rank
    assert len(logger.log_metrics.call_args[-1]["metrics"]) == 3 * world_size + 3
    assert "train/seconds_per_iter" in logger.log_metrics.call_args[-1]["metrics"]
    assert (
        "train/seconds_per_iter_averaged10"
        in logger.log_metrics.call_args[-1]["metrics"]
    )
    assert (
        "train/seconds_per_iter_averaged100"
        in logger.log_metrics.call_args[-1]["metrics"]
    )
    for i in range(world_size):
        assert (
            f"gpu_stats/utilization_rank{i}"
            in logger.log_metrics.call_args[-1]["metrics"]
        )
        assert (
            f"gpu_stats/max_memory_rank{i}"
            in logger.log_metrics.call_args[-1]["metrics"]
        )
        assert (
            f"gpu_stats/utilization_rank{i}_averaged10"
            in logger.log_metrics.call_args[-1]["metrics"]
        )

    for i in range(100):
        callback.on_train_batch_start(trainer, module, None, i + 13)

    assert callback.last_batch_start_time is not None
    assert len(callback.gpu_utilizations10) == world_size
    assert len(callback.gpu_utilizations100) == world_size
    for i in range(world_size):
        assert len(callback.gpu_utilizations10[i].sliding_window) == 10
        assert len(callback.gpu_utilizations100[i].sliding_window) == 100

    assert logger.log_metrics.call_count == 112
    # three times seconds per iter (current, averaged10, averaged100) + utilization per rank + max memory per rank
    # + utilization averaged10 per rank + utilization averaged100 per rank
    assert len(logger.log_metrics.call_args[-1]["metrics"]) == 4 * world_size + 3
    assert "train/seconds_per_iter" in logger.log_metrics.call_args[-1]["metrics"]
    assert (
        "train/seconds_per_iter_averaged10"
        in logger.log_metrics.call_args[-1]["metrics"]
    )
    assert (
        "train/seconds_per_iter_averaged100"
        in logger.log_metrics.call_args[-1]["metrics"]
    )
    for i in range(world_size):
        assert (
            f"gpu_stats/utilization_rank{i}"
            in logger.log_metrics.call_args[-1]["metrics"]
        )
        assert (
            f"gpu_stats/max_memory_rank{i}"
            in logger.log_metrics.call_args[-1]["metrics"]
        )
        assert (
            f"gpu_stats/utilization_rank{i}_averaged10"
            in logger.log_metrics.call_args[-1]["metrics"]
        )
        assert (
            f"gpu_stats/utilization_rank{i}_averaged100"
            in logger.log_metrics.call_args[-1]["metrics"]
        )

    print("")


@pytest.mark.parametrize("world_size", [1, 2, 4, 8])
def test_custom_monitoring_callback_train(world_size):
    torch.multiprocessing.spawn(
        _custom_monitoring_callback_train_mock, args=(world_size,), nprocs=world_size
    )


@pytest.mark.parametrize("world_size", [1, 2, 4, 42])
def test_custom_monitoring_checkpoint(world_size):
    trainer = mock.MagicMock()
    trainer.world_size = world_size
    cb = CustomMonitoringCallback()
    cb._init_gpu_util_trackers(trainer.world_size)
    ckpt = {}
    cb.on_save_checkpoint(mock.MagicMock(), mock.MagicMock(), ckpt)
    assert (
        len(ckpt) == 4
    )  # gpu_utilizations10, gpu_utilizations100, seconds_per_iter10, seconds_per_iter100

    assert len(ckpt["gpu_utilizations10"]) == world_size
    assert len(ckpt["gpu_utilizations100"]) == world_size

    cb2 = CustomMonitoringCallback()
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
