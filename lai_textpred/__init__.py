from lai_textpred.callbacks import default_callbacks  # noqa: F401
from lai_textpred.configs import *  # noqa: F401, F403
from lai_textpred.configs import __all__ as __all_configs
from lai_textpred.dataset import WordDataset  # noqa: F401
from lai_textpred.error import error_if_local  # noqa: F401

__version__ = "0.0.1"

__all__ = ["default_callbacks", "WordDataset", "error_if_local"] + __all_configs
