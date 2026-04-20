from . import Epidemics
from . import Dynamic
from . import DistributedComputing
from . import Opinions
from . import InfluenceMaximization
from . import utils


__all__ = [
    "Epidemics",
    "Dynamic",
    "DistributedComputing",
    "Opinions",
    "InfluenceMaximization",
    "utils",
]


from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("fs-gplib")
except PackageNotFoundError:
    __version__ = "0.0.0"