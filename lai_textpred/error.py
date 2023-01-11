from lightning.app.utilities.cloud import is_running_in_cloud


def error_if_local():
    if is_running_in_cloud():
        return

    raise RuntimeError(
        "This app is optimized for cloud usage running very large models. "
        "To run a similar app locally, please refer to "
        "https://github.com/Lightning-AI/LAI-Text-Classification-Component#running-locally-limited"
    )
