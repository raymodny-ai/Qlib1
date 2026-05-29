"""
LightGBM 梯度提升模型 (自包含模块)

专为表格型基本面因子数据优化，天然支持缺失值处理
和类别特征，在金融横截面预测中表现优秀。

PRD 2.2 要求: fit/predict 统一接口 + Rank IC 自动验证。

参数:
    objective: 'regression' | 'binary' | 'lambdarank' (默认 'regression')
    boosting_type: 'gbdt' | 'dart' | 'goss' (默认 'gbdt')
    num_leaves: 叶节点数 (默认 64)
    learning_rate: 学习率 (默认 0.05)
    n_estimators: 树数量 (默认 1000)
    early_stopping_rounds: 早停轮数 (默认 50)
    subsample: 样本采样率 (默认 0.8)
    colsample_bytree: 特征采样率 (默认 0.8)
    reg_alpha: L1 正则化 (默认 0.1)
    reg_lambda: L2 正则化 (默认 1.0)
    min_child_samples: 叶节点最小样本数 (默认 20)

使用示例:
    from src.models.lightgbm_model import LightGBMModel

    model = LightGBMModel(num_leaves=64, learning_rate=0.05)
    result = model.fit(X_train, y_train, X_valid, y_valid)
    preds = model.predict(X_test)
    # 带 Rank IC 的预测:
    preds = model.predict(X_test, y_true=y_test)
    print(f"Rank IC: {preds.rank_ic:.4f}")
"""

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
from scipy.stats import spearmanr

from src.analyzers.ml_pipeline import BaseForecastModel, PredictionResult, TrainingResult
from src.utils.logger import get_logger


def _compute_rank_ic(predictions: np.ndarray, y_true: np.ndarray) -> float:
    """计算 Rank IC (Spearman 相关系数)"""
    if predictions.ndim > 1:
        predictions = predictions.flatten()
    if y_true.ndim > 1:
        y_true = y_true.flatten()
    mask = ~(np.isnan(predictions) | np.isnan(y_true))
    if mask.sum() < 10:
        return 0.0
    ic, _ = spearmanr(predictions[mask], y_true[mask])
    return float(ic) if not np.isnan(ic) else 0.0


class LightGBMModel(BaseForecastModel):
    """
    LightGBM 梯度提升模型

    专为表格型基本面因子数据优化，天然支持缺失值处理
    和类别特征，在金融横截面预测中表现优秀。
    """

    def __init__(self, **params):
        defaults = {
            "objective": "regression",
            "boosting_type": "gbdt",
            "num_leaves": 64,
            "learning_rate": 0.05,
            "n_estimators": 1000,
            "early_stopping_rounds": 50,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.1,
            "reg_lambda": 1.0,
            "min_child_samples": 20,
            "verbose": -1,
            "random_state": 42,
            "n_jobs": -1,
        }
        defaults.update(params)
        super().__init__(**defaults)

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_valid: Optional[np.ndarray] = None,
        y_valid: Optional[np.ndarray] = None,
        feature_names: Optional[List[str]] = None,
    ) -> TrainingResult:
        try:
            import lightgbm as lgb
        except ImportError:
            raise ImportError(
                "LightGBM 未安装。请执行: pip install lightgbm"
            )

        self._feature_names = feature_names or [f"f{i}" for i in range(X_train.shape[1])]

        start = time.time()

        # 构建数据集
        train_data = lgb.Dataset(X_train, label=y_train)
        valid_sets = [train_data]
        valid_names = ["train"]

        if X_valid is not None and y_valid is not None:
            valid_data = lgb.Dataset(X_valid, label=y_valid, reference=train_data)
            valid_sets.append(valid_data)
            valid_names.append("valid")

        # 训练参数
        fit_params = self.params.copy()
        early_stopping = fit_params.pop("early_stopping_rounds", 50)
        n_estimators = fit_params.pop("n_estimators", 1000)
        random_state = fit_params.pop("random_state", 42)

        # 设置早停指标
        if fit_params.get("objective") in ("binary", "multiclass"):
            metric = "auc"
        else:
            metric = "rmse"

        callbacks = []
        if len(valid_sets) > 1:
            callbacks.append(lgb.early_stopping(early_stopping))
            callbacks.append(lgb.log_evaluation(period=0))

        # 执行训练
        self._model = lgb.train(
            fit_params,
            train_data,
            num_boost_round=n_estimators,
            valid_sets=valid_sets,
            valid_names=valid_names,
            callbacks=callbacks,
        )

        self._fitted = True
        train_time = (time.time() - start) * 1000

        # 提取训练/验证损失
        train_loss = []
        valid_loss = []
        try:
            eval_result = self._model.best_score
            if "train" in eval_result:
                train_loss = eval_result["train"].get(metric, [])
            if "valid" in eval_result:
                valid_loss = eval_result["valid"].get(metric, [])
        except Exception:
            pass

        best_iter = self._model.best_iteration if hasattr(self._model, "best_iteration") else n_estimators

        # 特征重要性
        importance = self.get_feature_importance()

        # 验证集 Rank IC
        rank_ic = 0.0
        if X_valid is not None and y_valid is not None:
            valid_preds = self._model.predict(X_valid)
            rank_ic = _compute_rank_ic(valid_preds, y_valid)

        result = TrainingResult(
            model_name="LightGBM",
            train_loss=train_loss if isinstance(train_loss, list) else [],
            valid_loss=valid_loss if isinstance(valid_loss, list) else [],
            best_iteration=best_iter,
            best_score=self._model.best_score.get("valid") if hasattr(self._model, "best_score") else 0,
            feature_importance=importance,
            train_time_ms=train_time,
            n_features=X_train.shape[1],
            n_samples=X_train.shape[0],
        )

        self.logger.info(
            "LightGBM 训练完成",
            n_estimators_used=best_iter,
            train_ms=round(train_time, 0),
            n_features=X_train.shape[1],
            rank_ic=round(rank_ic, 4),
        )

        return result

    def predict(self, X: np.ndarray, y_true: Optional[np.ndarray] = None) -> PredictionResult:
        """
        生成预测 (可选 Rank IC 自动验证)

        Args:
            X: 特征矩阵
            y_true: 真实标签 (可选, 提供时自动计算 Rank IC)

        Returns:
            PredictionResult (含 rank_ic 属性当 y_true 可用时)
        """
        if not self._fitted or self._model is None:
            raise RuntimeError("模型尚未训练，请先调用 fit()")

        preds = self._model.predict(X)
        predictions = preds if preds.ndim == 1 else preds.flatten()

        result = PredictionResult(predictions=predictions)

        if y_true is not None:
            rank_ic = _compute_rank_ic(predictions, y_true)
            result.rank_ic = rank_ic
            result.rank_icir = rank_ic / (predictions.std() / max(predictions.std(), 1e-12))

        return result

    def get_feature_importance(self) -> Dict[str, float]:
        if self._model is None:
            return {}

        try:
            names = self._model.feature_name()
            gains = self._model.feature_importance(importance_type="gain")
        except Exception:
            return {}

        total = sum(gains) or 1.0
        return {
            name: round(gain / total, 6)
            for name, gain in sorted(
                zip(names, gains), key=lambda x: x[1], reverse=True
            )
        }
