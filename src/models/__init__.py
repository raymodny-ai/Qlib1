"""
ML 模型模块

按 PRD 架构规范存放所有预测模型，统一继承自
src.analyzers.ml_pipeline.BaseForecastModel。

包含:
- LightGBMModel: LightGBM 梯度提升模型 (GBDT/DART/GOSS)
- XGBoostModel: XGBoost 梯度提升模型
- AdaRNNModel: 自适应循环神经网络 (GRU + MMD)
- TabNetModel: 注意力表格学习模型 (pytorch-tabnet)
- DoubleEnsembleModel: 双层加权集成模型

模型工厂:
    通过 BaseForecastModel.MODEL_REGISTRY 自动注册，
    支持从 YAML 配置动态实例化 (如 qlib_config.yaml 中的 models.registry)。
"""

from src.analyzers.ml_pipeline import BaseForecastModel
from src.models.lightgbm_model import LightGBMModel
from src.models.xgboost_model import XGBoostModel
from src.models.adarnn_model import AdaRNNModel
from src.models.tabnet_model import TabNetModel
from src.models.double_ensemble_model import DoubleEnsembleModel

# ---- 模型工厂 (YAML 驱动) ----

_MODEL_FACTORY: dict = {
    "lightgbm": LightGBMModel,
    "xgboost": XGBoostModel,
    "adarnn": AdaRNNModel,
    "tabnet": TabNetModel,
    "double_ensemble": DoubleEnsembleModel,
}


def create_model(model_key: str, **kwargs) -> BaseForecastModel:
    """
    从 YAML 配置中的模型 key 动态创建模型实例。

    优先使用 _MODEL_FACTORY 精确匹配，
    降级到 BaseForecastModel.MODEL_REGISTRY 模糊匹配。

    Args:
        model_key: 模型标识符 (如 'lightgbm', 'xgboost', 'double_ensemble')
        **kwargs: 模型超参数

    Returns:
        BaseForecastModel 子类实例

    Raises:
        ValueError: 未知模型 key
    """
    # 精确匹配
    cls = _MODEL_FACTORY.get(model_key.lower())
    if cls is not None:
        return cls(**kwargs)
    # 降级: 使用基类注册表
    return BaseForecastModel.create(model_key, **kwargs)


def list_available_models() -> list:
    """列出所有可用模型名称"""
    return list(_MODEL_FACTORY.keys())


__all__ = [
    "BaseForecastModel",
    "LightGBMModel",
    "XGBoostModel",
    "AdaRNNModel",
    "TabNetModel",
    "DoubleEnsembleModel",
    "create_model",
    "list_available_models",
]
