from .VoterModel import VoterModel
from .QVoterModel import QVoterModel
from .MajorityRuleModel import MajorityRuleModel
from .SznajdModel import SznajdModel

# HK / WHK 依赖 torch_scatter；惰性导入，避免未安装时无法 import Opinions 其余模型


def __getattr__(name):
    if name == "HKModel":
        from .HKModel import HKModel

        return HKModel
    if name == "WHKModel":
        from .WHKModel import WHKModel

        return WHKModel
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "VoterModel",
    "QVoterModel",
    "MajorityRuleModel",
    "SznajdModel",
    "HKModel",
    "WHKModel",
]
