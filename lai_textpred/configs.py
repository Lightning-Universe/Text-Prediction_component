__all__ = ["gpt_1_7b", "gpt_2_9b", "gpt_4_4b", "gpt_8_4b", "gpt_10b" ,"gpt_20b", "gpt_45b"]

from typing import Optional


def _build_gpt_config(
    n_layer: Optional[int] = None,
    n_head: Optional[int] = None,
    n_embed: Optional[int] = None,
    model_type: Optional[str] = None,
):
    if model_type is not None:
        assert n_layer is None and n_head is None and n_embed is None

    if n_layer is not None or n_head is not None or n_embed is not None:
        assert model_type is None
        assert n_layer is not None and n_head is not None and n_embed is not None

    return {"n_layer": n_layer, "n_head": n_head, "n_embd": n_embed, "model_type": model_type}


# all configs taken from https://github.com/SeanNaren/minGPT#training-billion-parameter-gpt-models
gpt_1_7b = _build_gpt_config(
    15, 16, 3072, None
)  # requires 2GB per GPU on 8 GPUs, 5.1 GB on 1 GPU

gpt_2_9b = _build_gpt_config(
    None, None, None, "gpt2_xxl"
)

gpt_4_4b = _build_gpt_config(
    None, None, None, "gpt2_xxxl"
)

gpt_8_4b = _build_gpt_config(
    None, None, None, 'gpt2_4xl'
)

gpt_10b = _build_gpt_config(
    13, 16, 8192, None
)  # requires 6GB per GPU on 8 GPUs, 26GB on 1 GPU

gpt_20b = _build_gpt_config(
    25, 16, 8192, None
)  # requires 8GB per GPU on 8 GPUs and ~500 GB of CPU RAM

gpt_45b = _build_gpt_config(
    56, 16, 8192, None
)  # requires 14 GB per GPU on 8 GPUs and ~950 GB of CPU RAM
