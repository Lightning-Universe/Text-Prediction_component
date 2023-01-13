from typing import cast
from unittest.mock import MagicMock

import lightning.pytorch.strategies
import pytest
import torch
from lightning_gpt import DeepSpeedNanoGPT

from lai_textpred.callbacks.steady_state_detection import SteadyStateDetection
from lai_textpred.steady_state_utils import chinchilla_metric_samples
from tests.helpers import setup_ddp


def test_steady_state_warning():
    with pytest.warns(UserWarning, match="Cuda utilization as a proxy metric"):
        SteadyStateDetection(target_loss=0.1, steady_state_det_mode="utilization")


def test_steady_state_error():
    with pytest.raises(ValueError, match="steady_state_det_mode must be either"):
        SteadyStateDetection(target_loss=0.1, steady_state_det_mode="invalid")


def test_num_samples_required():
    assert SteadyStateDetection(
        target_loss=0.1, num_params=100
    ).num_samples_required == chinchilla_metric_samples(0.1, 100)

    with pytest.raises(ValueError, match="Cannot calculate the number of samples"):
        _ = SteadyStateDetection(target_loss=0.1).num_samples_required


@pytest.mark.parametrize(
    "strategy_cls",
    [
        lightning.pytorch.strategies.DeepSpeedStrategy,
        lightning.pytorch.strategies.DDPStrategy,
        lightning.pytorch.strategies.SingleDeviceStrategy,
    ],
)
def test_steady_state_param_detection(strategy_cls):
    cb = SteadyStateDetection(target_loss=0.1)
    model = DeepSpeedNanoGPT(model_type="gpt-nano", vocab_size=1, block_size=1)
    trainer = MagicMock()
    trainer.strategy = cast(strategy_cls, MagicMock())

    cb.on_train_batch_start(trainer, model, None, 0)
    assert cb.num_params == 0

    cb = SteadyStateDetection(target_loss=0.1)
    model.configure_sharded_model()
    cb.on_train_batch_start(trainer, model, None, 1)

    assert cb.num_params == 85056


@pytest.mark.parametrize("batch_size", [1, 5, 10, 100])
def test_steady_state_batchsize_detection(batch_size):
    trainer = MagicMock()
    trainer.strategy = MagicMock()
    trainer.strategy.root_device = torch.device("cpu")
    trainer.strategy.reduce = lambda x, **kwargs: x
    cb = SteadyStateDetection(target_loss=0.1)
    cb.on_train_batch_end(trainer, MagicMock(), None, torch.rand(batch_size, 1), 0)
    assert cb.batch_size == batch_size

    cb = SteadyStateDetection(target_loss=0.1)
    cb.on_train_batch_end(
        trainer,
        MagicMock(),
        None,
        (torch.rand(batch_size, 1), torch.rand((batch_size, 5, 10))),
        0,
    )
    assert cb.batch_size == batch_size


def _test_steady_state_should_stop(
    rank,
    should_stop,
    num_steps_after_steady_state,
    world_size,
    gpu_util_logname,
    time_per_batch_logname,
):
    setup_ddp(rank, world_size)
    cb = SteadyStateDetection(
        target_loss=0.1,
        num_params=10,
        stop_on_steady_state=should_stop,
        steady_state_steps_before_stop=num_steps_after_steady_state,
        gpu_util_logname=gpu_util_logname,
        time_per_batch_logname=time_per_batch_logname,
    )

    trainer = MagicMock()
    trainer.strategy = MagicMock()
    trainer.strategy.root_device = torch.device("cpu")
    trainer.global_rank = rank
    trainer.world_size = world_size
    trainer.strategy.reduce = lambda x, **kwargs: x
    trainer.callback_metrics = {
        time_per_batch_logname: 0.1,
        f"{time_per_batch_logname}_averaged10": 0.1,
    }
    for j in range(world_size):
        trainer.callback_metrics.update(
            {
                f"{gpu_util_logname}_rank{j}": 0.1,
                f"{gpu_util_logname}_rank{j}_averaged10": 0.1,
            }
        )

    for i in range(9):
        cb.on_train_batch_end(trainer, MagicMock(), None, torch.rand(1, 1), i)
        assert cb.steady_state_achieved == (i >= 9)
        assert cb.steady_state_stepped == 0

    cb.steady_state_achieved = True
    for i in range(10):
        cb.on_train_batch_end(trainer, MagicMock(), None, torch.rand(1, 1), i + 10)
        assert cb.steady_state_achieved
        assert cb.steady_state_stepped == i + 1

        if should_stop and i > num_steps_after_steady_state:
            assert trainer.should_stop


@pytest.mark.parametrize("should_stop", [True, False])
@pytest.mark.parametrize("num_steps_after_steady_state", [1, 3])
@pytest.mark.parametrize("world_size", [1, 2, 4])
@pytest.mark.parametrize("gpu_util_logname", ["gpu_util", "foo"])
@pytest.mark.parametrize("time_per_batch_logname", ["time_per_batch", "bar"])
def test_steady_state_should_stop(
    should_stop,
    num_steps_after_steady_state,
    world_size,
    gpu_util_logname,
    time_per_batch_logname,
):
    torch.multiprocessing.spawn(
        _test_steady_state_should_stop,
        args=(
            should_stop,
            num_steps_after_steady_state,
            world_size,
            gpu_util_logname,
            time_per_batch_logname,
        ),
        nprocs=world_size,
    )
