"""
LightGBM 梯度提升模型 (独立模块)

从 src.analyzers.ml_pipeline 中分离为独立文件，
以支持 src.models 包的统一模型发现机制。

LightGBM 专为表格型基本面因子数据优化，天然支持缺失值处理
和类别特征，在金融横截面预测中表现优秀。

参数:
    loss: 损失函数类型 'mse' | 'mae' | 'huber' (默认 'mse')
    num_leaves: 叶节点数 (默认 64)
    max_depth: 树最大深度 (默认 8)
    learning_rate: 学习率 (默认 0.05)
    n_estimators: 树数量 (默认 500)
    subsample: 样本采样率 (默认 0.8)
    colsample_bytree: 特征采样率 (默认 0.8)
    reg_alpha: L1 正则化 (默认 0.1)
    reg_lambda: L2 正则化 (默认 0.1)
    early_stopping_rounds: 早停轮数 (默认 50)

使用示例:
    from src.models.lightgbm_model import LightGBMModel

    model = LightGBMModel(num_leaves=64, learning_rate=0.05)
    model.fit(X_train, y_train, X_valid, y_valid)
    predictions = model.predict(X_test)
"""

from src.analyzers.ml_pipeline import LightGBMModel as _LightGBMModel

# 重新导出，保持 API 兼容
LightGBMModel = _LightGBMModel

__all__ = ["LightGBMModel"]
