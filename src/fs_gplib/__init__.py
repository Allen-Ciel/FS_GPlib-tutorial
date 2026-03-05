from . import Epidemics
from . import Dynamic
from . import DistributedComputing
from . import Opinions
from . import utils

# 对外暴露的主要子模块
__all__ = [
    "Epidemics",
    "Dynamic",
    "DistributedComputing",
    "Opinions",
    "utils",
]


from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("fs-gplib")
except PackageNotFoundError:
    __version__ = "0.0.0"