from .core import *
from .api_models import *
from .roboreward import *
from .sole import *
from .topreward import *

__all__ = []

try:
    from importlib.metadata import version
    __version__ = version("rewardgen")
except Exception:
    __version__ = "0.1.4"

