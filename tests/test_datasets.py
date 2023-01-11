import math

import pytest
import requests
import torch

from lai_textpred.dataset import WordDataset


@pytest.mark.parametrize("block_size", [5, 10, 20])
def test_word_dataset(block_size):
    text = requests.get(
        "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    ).text
    dataset = WordDataset(text, block_size)
    assert dataset.vocab_size == 12849
    assert dataset.block_size == block_size
    assert len(dataset.data) == 209892
    assert len(dataset) == math.ceil(len(dataset.data) / (block_size + 1))

    assert dataset.stoi[""] == 0
    assert dataset.stoi["the"] == 1

    assert dataset.itos[0] == ""
    assert dataset.itos[1] == "the"

    assert isinstance(dataset[0], tuple)
    assert len(dataset[0]) == 2
    assert isinstance(dataset[0][0], torch.Tensor)
    assert isinstance(dataset[0][1], torch.Tensor)
    assert dataset[0][0].shape == (block_size,)
    assert dataset[0][1].shape == (block_size,)
    assert dataset[0][0].dtype == torch.int64
    assert dataset[0][1].dtype == torch.int64


def test_remove_punctuation():
    text_no_punct = WordDataset.remove_punctuation(
        "Hello! My Name is Foo; I'm a Bar. Who are you? - I'm a Bar too, named: Faa"
    )
    assert (
        text_no_punct
        == "Hello My Name is Foo Im a Bar Who are you  Im a Bar too named Faa"
    )
