__all__ = ["gpt_1_7b", "gpt_10b", "gpt_20b", "gpt_45b"]


def _build_gpt_config(
    n_layer: int,
    n_head: int,
    n_embed: int,
):
    return {"n_layer": n_layer, "n_head": n_head, "n_embd": n_embed}


# all configs taken from https://github.com/SeanNaren/minGPT#training-billion-parameter-gpt-models
gpt_1_7b = _build_gpt_config(15, 16, 3072)  # requires 2GB per GPU on 8 GPUs, 5.1 GB on 1 GPU
gpt_10b = _build_gpt_config(13, 16, 8192)  # requires 6GB per GPU on 8 GPUs, 26GB on 1 GPU
gpt_20b = _build_gpt_config(25, 16, 8192)  # requires 8GB per GPU on 8 GPUs and ~500 GB of CPU RAM
gpt_45b = _build_gpt_config(56, 16, 8192)  # requires 14 GB per GPU on 8 GPUs and ~950 GB of CPU RAM
