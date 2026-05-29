"""
XGBoost 梯度提升模型 (自包含模块)

XGBoost 提供高效的正则化梯度提升，在结构化表格数据上
表现稳定，广泛用于金融因子预测场景。

PRD 2.2 要求: fit/predict 统一接口 + Rank IC 自动验证。

参数:
    objective: 目标函数 'reg:squarederror' | 'reg:logistic' | 'binary:logistic'
    max_depth: 树深度 (默认 6)
    learning_rate: 学习率 (默认 0.05)
    n_estimators: 树数量 (默认 1000)
    early_stopping_rounds: 早停轮数 (默认 50)
    subsample: 样本采样率 (默认 0.8)
    colsample_bytree: 特征采样率 (默认 0.8)
    reg_alpha: L1 正则化 (默认 0.1)
    reg_lambda: L2 正则化 (默认 1.0)

使用示例:
    from src.models.xgboost_model import XGBoostModel

    model = XGBoostModel(max_depth=6, learning_rate=0.05)
    result = model.fit(X_train, y_train, X_valid, y_valid)
    preds = model.predict(X_test, y_true=y_test)
    print(f"Rank IC: {preds.rank_ic:.4f}")
"""

import time
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


class XGBoostModel(BaseForecastModel):
    """
    XGBoost 梯度提升模型

    提供高效的正则化梯度提升，在结构化表格数据上
    表现稳定，广泛用于金融因子预测场景。
    """

    def __init__(self, **params):
        defaults = {
            "objective": "reg:squarederror",
            "max_depth": 6,
            "learning_rate": 0.05,
            "n_estimators": 1000,
            "early_stopping_rounds": 50,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.1,
            "reg_lambda": 1.0,
            "verbosity": 0,
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
            import xgboost as xgb
        except ImportError:
            raise ImportError(
                "XGBoost 未安装。请执行: pip install xgboost"
            )

        self._feature_names = feature_names or [f"f{i}" for i in range(X_train.shape[1])]

        start = time.time()

        fit_params = self.params.copy()
        early_stopping = fit_params.pop("early_stopping_rounds", 50)
        n_estimators = fit_params.pop("n_estimators", 1000)
        random_state = fit_params.pop("random_state", 42)

        eval_set = [(X_train, y_train)]
        if X_valid is not None and y_valid is not None:
            eval_set.append((X_valid, y_valid))

        self._model = xgb.XGBRegressor(
            n_estimators=n_estimators,
            random_state=random_state,
            **{k: v for k, v in fit_params.items()
               if k not in ("n_estimators", "early_stopping_rounds")},
        )

        self._model.fit(
            X_train, y_train,
            eval_set=eval_set,
            verbose=False,
        )

        self._fitted = True
        train_time = (time.time() - start) * 1000

        # 提取训练/验证损失
        train_loss = []
        valid_loss = []
        evals_result = self._model.evals_result()
        if "validation_0" in evals_result:
            metric_key = list(evals_result["validation_0"].keys())[0]
            train_loss = evals_result["validation_0"][metric_key]
        if "validation_1" in evals_result:
            metric_key = list(evals_result["validation_1"].keys())[0]
            valid_loss = evals_result["validation_1"][metric_key]

        best_iter = getattr(self._model, "best_iteration", n_estimators)
        best_score = valid_loss[-1] if valid_loss else float("inf")

        importance = self.get_feature_importance()

        # 验证集 Rank IC
        rank_ic = 0.0
        if X_valid is not None and y_valid is not None:
            valid_preds = self._model.predict(X_valid)
            rank_ic = _compute_rank_ic(valid_preds, y_valid)

        result = TrainingResult(
            model_name="XGBoost",
            train_loss=train_loss if isinstance(train_loss, list) else [],
            valid_loss=valid_loss if isinstance(valid_loss, list) else [],
            best_iteration=best_iter,
            best_score=best_score,
            feature_importance=importance,
            train_time_ms=train_time,
            n_features=X_train.shape[1],
            n_samples=X_train.shape[0],
        )

        self.logger.info(
            "XGBoost 训练完成",
            best_iter=best_iter,
            train_ms=round(train_time, 0),
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
            gains = self._model.feature_importances_
            total = sum(gains) or 1.0
            return {
                name: round(gain / total, 6)
                for name, gain in sorted(
                    zip(self._feature_names, gains), key=lambda x: x[1], reverse=True
                )
            }
        except Exception:
            return {}
