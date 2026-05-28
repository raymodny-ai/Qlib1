"""
DoubleEnsemble 双层集成模型

基于 "DoubleEnsemble: A New Ensemble Method for Quantitative Trading" 思想，
通过加权平均集成多个基础模型 (LightGBM / XGBoost / TabNet / AdaRNN)。

核心思想:
- Layer 1: 多个异构基础模型独立训练
- Layer 2: 基于验证集 ICIR 动态分配权重，加权平均集成
- 支持静态等权重 / 动态 ICIR 权重 / 用户自定义权重

架构:
  Input → [LightGBM, XGBoost, TabNet, AdaRNN] → Weighted Average → Output

适用场景:
- 多模型风险分散
- 提升预测稳健性 (降低单模型过拟合)
- 因子预测的集成学习

设计原则:
- 继承 BaseForecastModel，遵循 fit/predict 统一接口
- 任意数量基础模型的灵活组合
- 自动处理模型异质性 (CPU/GPU, tree-based/deep)

使用示例:
    from src.analyzers.double_ensemble_model import DoubleEnsembleModel
    from src.analyzers.ml_pipeline import LightGBMModel, XGBoostModel
    from src.analyzers.tabnet_model import TabNetModel

    ensemble = DoubleEnsembleModel(
        base_models=[
            LightGBMModel(num_leaves=64),
            XGBoostModel(max_depth=6),
            TabNetModel(input_dim=200),
        ],
        weight_method="icir",
    )
    ensemble.fit(X_train, y_train, X_valid, y_valid)
    predictions = ensemble.predict(X_test)
"""

import os
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from src.analyzers.ml_pipeline import BaseForecastModel, TrainingResult, PredictionResult
from src.utils.logger import get_logger


class DoubleEnsembleModel(BaseForecastModel):
    """
    DoubleEnsemble: 双层加权集成模型

    集成多个异构基础模型，按验证集表现动态分配权重。

    参数:
        base_models: 基础模型列表 (BaseForecastModel 实例)
        weight_method: 权重方案
            - 'equal': 等权重平均
            - 'icir': 基于验证集 ICIR 动态权重
            - 'performance': 基于验证损失倒数权重
            - list[float]: 自定义权重列表
        use_oof: 是否使用 out-of-fold 预测 (默认 False)
        ensemble_mode: 'mean' | 'median' | 'weighted' (默认 'weighted')
    """

    def __init__(self, **params):
        defaults = {
            "base_models": None,
            "weight_method": "equal",  # 'equal' | 'icir' | 'performance' | list
            "use_oof": False,
            "ensemble_mode": "weighted",  # 'mean' | 'median' | 'weighted'
            "random_state": 42,
        }
        defaults.update(params)
        super().__init__(**defaults)

        self._base_models: List[BaseForecastModel] = (
            self.params["base_models"] or []
        )
        self._weights: np.ndarray = np.array([])
        self._model_names: List[str] = []
        self._individual_results: List[TrainingResult] = []
        self.logger = get_logger()

    # ================================================================
    #  fit
    # ================================================================

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_valid: Optional[np.ndarray] = None,
        y_valid: Optional[np.ndarray] = None,
        feature_names: Optional[List[str]] = None,
    ) -> TrainingResult:
        if not self._base_models:
            raise ValueError("至少需要一个基础模型。请在初始化时提供 base_models 参数。")

        self._feature_names = feature_names or [f"f{i}" for i in range(X_train.shape[1])]
        self._model_names = [m.__class__.__name__ for m in self._base_models]

        start_time = time.time()
        self._individual_results = []
        valid_predictions: List[np.ndarray] = []

        # ---- Layer 1: 逐模型训练 ----
        for i, model in enumerate(self._base_models):
            self.logger.info(f"训练基础模型 [{i+1}/{len(self._base_models)}]: {self._model_names[i]}")

            try:
                result = model.fit(
                    X_train=X_train,
                    y_train=y_train,
                    X_valid=X_valid,
                    y_valid=y_valid,
                    feature_names=feature_names,
                )
                self._individual_results.append(result)
            except Exception as e:
                self.logger.error(f"基础模型训练失败 [{self._model_names[i]}]: {e}")
                raise

            # 生成验证集预测
            if X_valid is not None and y_valid is not None:
                try:
                    pred_result = model.predict(X_valid)
                    valid_predictions.append(pred_result.predictions)
                except Exception:
                    valid_predictions.append(np.zeros(len(y_valid)))

        # ---- Layer 2: 计算权重 ----
        self._weights = self._calculate_weights(
            valid_predictions,
            y_valid,
        )

        self._fitted = True
        train_time_ms = (time.time() - start_time) * 1000

        # 聚合训练指标
        best_score = np.average(
            [r.best_score for r in self._individual_results],
            weights=None,
        ) if self._individual_results else float("inf")

        result = TrainingResult(
            model_name="DoubleEnsemble",
            train_loss=[],
            valid_loss=[],
            best_iteration=0,
            best_score=float(best_score),
            feature_importance=self.get_feature_importance(),
            train_time_ms=train_time_ms,
            n_features=X_train.shape[1],
            n_samples=X_train.shape[0],
        )

        self.logger.info(
            "DoubleEnsemble 训练完成",
            n_models=len(self._base_models),
            weights={n: round(float(w), 4) for n, w in zip(self._model_names, self._weights)},
            train_ms=round(train_time_ms, 0),
        )

        return result

    # ================================================================
    #  predict
    # ================================================================

    def predict(self, X: np.ndarray) -> PredictionResult:
        if not self._fitted:
            raise RuntimeError("模型尚未训练，请先调用 fit()")

        if not self._base_models:
            return PredictionResult(predictions=np.array([]))

        ensemble_mode = self.params["ensemble_mode"]
        predictions_list = []

        for i, model in enumerate(self._base_models):
            try:
                pred_result = model.predict(X)
                p = pred_result.predictions
                if p.ndim > 1:
                    p = p.flatten()
                predictions_list.append(p)
            except Exception as e:
                self.logger.warning(f"模型 [{self._model_names[i]}] 预测失败: {e}")
                predictions_list.append(np.zeros(X.shape[0]))

        n_models = len(predictions_list)

        if ensemble_mode == "median":
            stacked = np.column_stack(predictions_list)
            final = np.median(stacked, axis=1)
        elif ensemble_mode == "weighted":
            final = np.zeros(X.shape[0])
            for i, preds in enumerate(predictions_list):
                w = self._weights[i] if i < len(self._weights) else (1.0 / n_models)
                final += w * preds
        else:
            # 'mean'
            final = np.mean(np.column_stack(predictions_list), axis=1)

        return PredictionResult(predictions=final)

    # ================================================================
    #  权重计算
    # ================================================================

    def _calculate_weights(
        self,
        valid_predictions: List[np.ndarray],
        y_valid: Optional[np.ndarray],
    ) -> np.ndarray:
        """根据验证集表现计算集成权重"""
        n_models = len(self._base_models)
        weight_method = self.params["weight_method"]

        # 自定义权重
        if isinstance(weight_method, (list, tuple, np.ndarray)):
            weights = np.array(weight_method, dtype=float)
            if len(weights) != n_models:
                raise ValueError(f"自定义权重长度 {len(weights)} ≠ 模型数 {n_models}")
            return weights / weights.sum()

        # 等权重
        if weight_method == "equal":
            return np.ones(n_models) / n_models

        # 基于验证集计算
        if y_valid is None or not valid_predictions:
            self.logger.info("无验证集，使用等权重")
            return np.ones(n_models) / n_models

        if weight_method == "icir":
            return self._weights_by_icir(valid_predictions, y_valid)
        elif weight_method == "performance":
            return self._weights_by_performance(valid_predictions, y_valid)
        else:
            self.logger.warning(f"未知权重方案 '{weight_method}'，使用等权重")
            return np.ones(n_models) / n_models

    def _weights_by_icir(
        self,
        valid_predictions: List[np.ndarray],
        y_valid: np.ndarray,
    ) -> np.ndarray:
        """基于 ICIR (信息系数信息比率) 的权重"""
        icirs = []
        for preds in valid_predictions:
            preds = np.nan_to_num(preds, nan=0.0)
            y = np.nan_to_num(y_valid, nan=0.0)
            mask = (~np.isnan(preds)) & (~np.isnan(y)) & (~np.isinf(preds)) & (~np.isinf(y))

            if mask.sum() < 10:
                icirs.append(0.0)
                continue

            p = preds[mask]
            r = y[mask]

            try:
                ic = np.corrcoef(p, r)[0, 1]
                # 对短序列: ICIR ≈ IC (std≈1 近似)
                icir = abs(ic) if not np.isnan(ic) else 0.0
            except Exception:
                icir = 0.0

            icirs.append(icir)

        icirs = np.array(icirs)
        total = icirs.sum()

        if total <= 0:
            return np.ones(len(self._base_models)) / len(self._base_models)

        return icirs / total

    def _weights_by_performance(
        self,
        valid_predictions: List[np.ndarray],
        y_valid: np.ndarray,
    ) -> np.ndarray:
        """基于验证集 MSE 倒数的权重"""
        losses = []
        for preds in valid_predictions:
            preds = np.nan_to_num(preds, nan=0.0)
            y = np.nan_to_num(y_valid, nan=0.0)
            mask = (~np.isnan(preds)) & (~np.isnan(y)) & (~np.isinf(preds)) & (~np.isinf(y))

            if mask.sum() < 10:
                losses.append(float("inf"))
                continue

            mse = np.mean((preds[mask] - y[mask]) ** 2)
            losses.append(mse)

        # 倒数权重 (MSE 越小权重越大)
        inv_losses = 1.0 / (np.array(losses) + 1e-12)
        total = inv_losses.sum()

        if total <= 0:
            return np.ones(len(self._base_models)) / len(self._base_models)

        return inv_losses / total

    # ================================================================
    #  特征重要性
    # ================================================================

    def get_feature_importance(self) -> Dict[str, float]:
        """聚合加权特征重要性"""
        if not self._base_models:
            return {}

        aggregated: Dict[str, float] = {}
        n_models = len(self._base_models)

        for i, model in enumerate(self._base_models):
            w = self._weights[i] if i < len(self._weights) else (1.0 / n_models)
            imp = model.get_feature_importance()
            for name, score in imp.items():
                aggregated[name] = aggregated.get(name, 0.0) + w * score

        # 归一化
        total = sum(aggregated.values()) or 1.0
        return {k: round(v / total, 6) for k, v in sorted(
            aggregated.items(), key=lambda x: x[1], reverse=True
        )}

    # ================================================================
    #  保存 / 加载
    # ================================================================

    def save(self, path: str):
        """保存集成模型 (各基础模型独立保存)"""
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        import pickle

        base_path = path.replace(".pkl", "")
        base_model_paths = []

        for i, model in enumerate(self._base_models):
            model_path = f"{base_path}_base_{i}.pkl"
            model.save(model_path)
            base_model_paths.append(model_path)

        data = {
            "params": self.params,
            "feature_names": self._feature_names,
            "weights": self._weights.tolist() if len(self._weights) > 0 else [],
            "model_names": self._model_names,
            "base_model_paths": base_model_paths,
            "fitted": self._fitted,
        }
        with open(path, "wb") as f:
            pickle.dump(data, f)

        self.logger.info("DoubleEnsemble 模型已保存", path=path)

    def load(self, path: str):
        """加载集成模型"""
        import pickle

        with open(path, "rb") as f:
            data = pickle.load(f)

        self.params = data.get("params", self.params)
        self._feature_names = data.get("feature_names", [])
        self._weights = np.array(data.get("weights", []))
        self._model_names = data.get("model_names", [])
        self._fitted = data.get("fitted", False)

        base_model_paths = data.get("base_model_paths", [])
        for model_path in base_model_paths:
            if os.path.exists(model_path):
                try:
                    with open(model_path, "rb") as f2:
                        model_data = pickle.load(f2)
                    # 重建模型
                    model_cls = BaseForecastModel.MODEL_REGISTRY.get(
                        self._model_names[len(self._base_models)] if len(self._base_models) < len(self._model_names) else "lightgbm"
                    )
                    if model_cls is None:
                        from src.analyzers.ml_pipeline import LightGBMModel
                        model_cls = LightGBMModel
                    # 简单重建 (子类自行负责完整 load)
                except Exception:
                    continue

        # 尝试逐个加载基础模型
        for model_path in base_model_paths:
            if not os.path.exists(model_path):
                continue
            for model_cls_name in ["lightgbm", "xgboost", "tabnet", "adarnn"]:
                try:
                    model_cls = BaseForecastModel.MODEL_REGISTRY.get(model_cls_name)
                    if model_cls is None:
                        continue
                    model = model_cls()
                    model.load(model_path)
                    self._base_models.append(model)
                    break
                except Exception:
                    continue

        self.logger.info("DoubleEnsemble 模型已加载", path=path)

    # ================================================================
    #  便利属性
    # ================================================================

    @property
    def weights(self) -> Dict[str, float]:
        """模型权重的可读字典"""
        return {
            name: round(float(w), 4)
            for name, w in zip(self._model_names, self._weights)
        }

    @property
    def model_names(self) -> List[str]:
        return self._model_names

    @property
    def individual_results(self) -> List[TrainingResult]:
        return self._individual_results
