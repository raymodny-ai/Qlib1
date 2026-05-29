"""
XGBoost 梯度提升模型 (独立模块)

从 src.analyzers.ml_pipeline 中分离为独立文件，
以支持 src.models 包的统一模型发现机制。

XGBoost 提供高效的正则化梯度提升，在结构化表格数据上
表现稳定，广泛用于金融因子预测场景。

参数:
    objective: 目标函数 'reg:squarederror' | 'reg:logistic' (默认 'reg:squarederror')
    max_depth: 树深度 (默认 6)
    learning_rate: 学习率 (默认 0.05)
    n_estimators: 树数量 (默认 500)
    subsample: 样本采样率 (默认 0.8)
    colsample_bytree: 特征采样率 (默认 0.8)
    reg_alpha: L1 正则化 (默认 0.1)
    reg_lambda: L2 正则化 (默认 1.0)
    early_stopping_rounds: 早停轮数 (默认 50)

使用示例:
    from src.models.xgboost_model import XGBoostModel

    model = XGBoostModel(max_depth=6, learning_rate=0.05)
    model.fit(X_train, y_train, X_valid, y_valid)
    predictions = model.predict(X_test)
"""

from src.analyzers.ml_pipeline import XGBoostModel as _XGBoostModel

# 重新导出，保持 API 兼容
XGBoostModel = _XGBoostModel

__all__ = ["XGBoostModel"]
