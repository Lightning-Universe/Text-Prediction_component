import pytest
import torch

from lai_textpred.moving_average import MovingAverage


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