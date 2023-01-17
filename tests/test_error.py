import sys
from unittest import mock

import pytest


@mock.patch("lightning.app.utilities.cloud.is_running_in_cloud", return_value=False)
def test_error_appears_local(patch):
    sys.modules.pop("lai_textpred.error", None)
    from lai_textpred.error import error_if_local

    with pytest.raises(RuntimeError):
        error_if_local()


@mock.patch("lightning.app.utilities.cloud.is_running_in_cloud", return_value=True)
def test_error_does_not_appear_cloud(patch):
    sys.modules.pop("lai_textpred.error", None)
    from lai_textpred.error import error_if_local

    error_if_local()
