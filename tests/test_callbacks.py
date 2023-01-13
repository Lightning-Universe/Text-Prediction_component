import lightning

from lai_textpred.callbacks import GPUMonitoringCallback, default_callbacks


# TODO: check for arguments
def test_default_callbacks():
    assert isinstance(default_callbacks()[0], lightning.pytorch.callbacks.EarlyStopping)
    assert isinstance(
        default_callbacks()[1], lightning.pytorch.callbacks.ModelCheckpoint
    )
    assert isinstance(default_callbacks()[2], lightning.pytorch.callbacks.Callback)
    assert isinstance(default_callbacks()[2], GPUMonitoringCallback)
