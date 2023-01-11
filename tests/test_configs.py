import pytest
from lightning_gpt.models import MinGPT, NanoGPT

from lai_textpred.configs import gpt_1_7b


# only test 1_7b params since everything else can't easily be run locally
@pytest.mark.parametrize("config", [gpt_1_7b])
@pytest.mark.parametrize("model_cls", [MinGPT, NanoGPT])
def test_gpt_configs(model_cls, config):
    model = model_cls(vocab_size=100, block_size=100, model_type=None, **config)
    getattr(model, "configure_sharded_model", lambda: None)()
    num_params = sum(p.numel() for p in model.parameters())
    assert num_params
