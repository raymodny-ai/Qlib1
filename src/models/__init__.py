"""
深度学习模型模块

按 PRD 架构规范存放 AdaRNN、TabNet、DoubleEnsemble 等深度模型。
所有模型继承自 src.analyzers.ml_pipeline.BaseForecastModel。

包含:
- AdaRNNModel: 自适应循环神经网络 (GRU + MMD)
- TabNetModel: 注意力表格学习模型 (pytorch-tabnet)
- DoubleEnsembleModel: 双层加权集成模型
"""

from src.models.adarnn_model import AdaRNNModel
from src.models.tabnet_model import TabNetModel
from src.models.double_ensemble_model import DoubleEnsembleModel

__all__ = [
    "AdaRNNModel",
    "TabNetModel",
    "DoubleEnsembleModel",
]
