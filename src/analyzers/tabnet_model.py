"""
TabNet 深度表格模型

基于 "TabNet: Attentive Interpretable Tabular Learning" (AAAI 2021)
使用 pytorch-tabnet 的 TabNetRegressor 作为后端。

核心思想:
- Sequential Attention: 每步选择一部分特征进行推理
- Feature Transformer: 共享 + 决策步特定层
- Attentive Transformer: 基于前步输出的稀疏特征选择
- 天然具备特征重要性可解释性 (mask 权重)

架构:
  Input → Feature Transformer → Attentive Transformer → Split → FC Output
                                    ↑_________________________________↓

适用场景:
- 基本面因子横截面预测
- 高维稀疏特征 (200+ 因子)
- 需要特征选择可解释性的场景

设计原则:
- 继承 BaseForecastModel，遵循 fit/predict 统一接口
- 自动处理缺失值 (TabNet 原生支持)
- GPU 加速 (自动检测 CUDA)

使用示例:
    from src.analyzers.tabnet_model import TabNetModel

    model = TabNetModel(
        input_dim=200,
        n_d=32,
        n_a=32,
        n_steps=5,
        gamma=1.5,
        learning_rate=2e-2,
    )
    model.fit(X_train, y_train, X_valid, y_valid)
    predictions = model.predict(X_test)
"""

import os
import time
from typing import Any, Dict, List, Optional

import numpy as np

from src.analyzers.ml_pipeline import BaseForecastModel, TrainingResult, PredictionResult
from src.utils.logger import get_logger


class TabNetModel(BaseForecastModel):
    """
    TabNet: 注意力表格学习模型

    基于 pytorch-tabnet 的 TabNetRegressor，专为表格数据优化，
    具备稀疏特征选择和注意力可解释性。

    参数:
        input_dim: 输入特征维度 (必填)
        output_dim: 输出维度 (默认 1)
        n_d: 决策预测层宽度 (默认 32)
        n_a: 注意力嵌入宽度 (默认 32)
        n_steps: 决策步数 (默认 5)
        gamma: 特征复用系数 (默认 1.5)
        n_independent: 独立门控线性单元层数 (默认 2)
        n_shared: 共享门控线性单元层数 (默认 2)
        momentum: BatchNorm 动量 (默认 0.3)
        mask_type: 注意力掩码类型 'sparsemax' | 'entmax' (默认 'entmax')
        learning_rate: 学习率 (默认 2e-2)
        n_epochs: 最大训练轮数 (默认 300)
        batch_size: 批次大小 (默认 256)
        virtual_batch_size: 虚拟批次大小 (默认 128, Ghost BN)
        early_stopping_patience: 早停耐心轮数 (默认 30)
        scheduler_patience: 学习率衰减耐心 (默认 10)
        scheduler_factor: 学习率衰减因子 (默认 0.5)
    """

    def __init__(self, **params):
        defaults = {
            "input_dim": 200,
            "output_dim": 1,
            "n_d": 32,
            "n_a": 32,
            "n_steps": 5,
            "gamma": 1.5,
            "n_independent": 2,
            "n_shared": 2,
            "momentum": 0.3,
            "mask_type": "entmax",
            "learning_rate": 2e-2,
            "n_epochs": 300,
            "batch_size": 256,
            "virtual_batch_size": 128,
            "early_stopping_patience": 30,
            "scheduler_patience": 10,
            "scheduler_factor": 0.5,
            "random_state": 42,
        }
        defaults.update(params)
        super().__init__(**defaults)
        self._device = self._detect_device()
        self._feature_importance_dict: Dict[str, float] = {}
        self.logger = get_logger()

    @staticmethod
    def _detect_device() -> str:
        try:
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"

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
        try:
            from pytorch_tabnet.tab_model import TabNetRegressor
        except ImportError:
            raise ImportError(
                "pytorch-tabnet 未安装。请执行: pip install pytorch-tabnet>=4.1.0"
            )

        self._feature_names = feature_names or [f"f{i}" for i in range(X_train.shape[1])]

        params = self.params
        input_dim = params["input_dim"]
        output_dim = params["output_dim"]
        n_epochs = params["n_epochs"]
        batch_size = params["batch_size"]
        virtual_batch_size = params["virtual_batch_size"]
        patience = params["early_stopping_patience"]
        scheduler_patience = params["scheduler_patience"]
        scheduler_factor = params["scheduler_factor"]

        # 处理数据
        X_train = np.nan_to_num(X_train.astype(np.float32), nan=0.0)
        y_train = y_train.astype(np.float32).reshape(-1, 1)

        X_valid_arr = None
        y_valid_arr = None
        if X_valid is not None and y_valid is not None:
            X_valid_arr = np.nan_to_num(X_valid.astype(np.float32), nan=0.0)
            y_valid_arr = y_valid.astype(np.float32).reshape(-1, 1)

        # 构建模型
        tabnet_params = {
            "n_d": params["n_d"],
            "n_a": params["n_a"],
            "n_steps": params["n_steps"],
            "gamma": params["gamma"],
            "n_independent": params["n_independent"],
            "n_shared": params["n_shared"],
            "momentum": params["momentum"],
            "mask_type": params["mask_type"],
            "seed": params["random_state"],
        }

        self._model = TabNetRegressor(
            input_dim=input_dim,
            output_dim=output_dim,
            **tabnet_params,
        )

        start_time = time.time()
        train_losses = []
        valid_losses = []
        best_epoch = 0

        # TabNet 训练 (内置早停 & 学习率调度)
        self._model.fit(
            X_train=X_train,
            y_train=y_train,
            eval_set=[(X_valid_arr, y_valid_arr)] if X_valid_arr is not None else None,
            eval_name=["valid"] if X_valid_arr is not None else None,
            eval_metric=["mse"] if X_valid_arr is not None else None,
            max_epochs=n_epochs,
            patience=patience,
            batch_size=batch_size,
            virtual_batch_size=min(virtual_batch_size, len(X_train) // 2),
            num_workers=0,
            drop_last=False,
            # 学习率调度
            scheduler_params={
                "scheduler": "ReduceLROnPlateau",
                "mode": "min",
                "factor": scheduler_factor,
                "patience": scheduler_patience,
                "min_lr": 1e-6,
            },
        )

        # 提取损失历史
        try:
            history = self._model.history
            train_losses = history.get("loss", [])
            if history.get("valid_mse"):
                valid_losses = [v[0] if isinstance(v, list) else v for v in history["valid_mse"]]
        except Exception:
            pass

        train_time_ms = (time.time() - start_time) * 1000
        best_valid_loss = min(valid_losses) if valid_losses else (min(train_losses) if train_losses else float("inf"))

        # 特征重要性 (mask-based)
        self._feature_importance_dict = self._compute_feature_importance(X_train)

        self._fitted = True

        result = TrainingResult(
            model_name="TabNet",
            train_loss=train_losses,
            valid_loss=valid_losses,
            best_iteration=len(train_losses),
            best_score=best_valid_loss,
            feature_importance=self._feature_importance_dict,
            train_time_ms=train_time_ms,
            n_features=input_dim,
            n_samples=X_train.shape[0],
        )

        self.logger.info(
            "TabNet 训练完成",
            epochs=len(train_losses),
            best_loss=round(best_valid_loss, 6),
            train_ms=round(train_time_ms, 0),
        )

        return result

    # ================================================================
    #  predict
    # ================================================================

    def predict(self, X: np.ndarray) -> PredictionResult:
        if not self._fitted or self._model is None:
            raise RuntimeError("模型尚未训练，请先调用 fit()")

        X = np.nan_to_num(X.astype(np.float32), nan=0.0)
        preds = self._model.predict(X)

        return PredictionResult(predictions=preds if preds.ndim == 1 else preds.flatten())

    # ================================================================
    #  特征重要性
    # ================================================================

    def _compute_feature_importance(self, X_sample: np.ndarray) -> Dict[str, float]:
        """通过 TabNet feature_importances_ 计算特征重要性"""
        try:
            importances = self._model.feature_importances_
            if importances is None:
                return {}

            total = sum(importances) or 1.0
            result = {}
            for i, imp in enumerate(importances):
                name = self._feature_names[i] if i < len(self._feature_names) else f"f{i}"
                result[name] = round(imp / total, 6)

            # 按重要性排序
            return dict(sorted(result.items(), key=lambda x: x[1], reverse=True))
        except Exception:
            return {}

    def get_feature_importance(self) -> Dict[str, float]:
        return self._feature_importance_dict

    # ================================================================
    #  保存 / 加载
    # ================================================================

    def save(self, path: str):
        """保存 TabNet 模型"""
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        import pickle

        data = {
            "params": self.params,
            "feature_names": self._feature_names,
            "feature_importance": self._feature_importance_dict,
            "fitted": self._fitted,
        }
        with open(path, "wb") as f:
            pickle.dump(data, f)

        # 保存 TabNet 原生权重
        weight_path = path.replace(".pkl", ".zip")
        if hasattr(self._model, "save_model"):
            self._model.save_model(weight_path)

        self.logger.info("TabNet 模型已保存", path=path)

    def load(self, path: str):
        """加载 TabNet 模型"""
        import pickle
        from pytorch_tabnet.tab_model import TabNetRegressor

        with open(path, "rb") as f:
            data = pickle.load(f)

        self.params = data.get("params", self.params)
        self._feature_names = data.get("feature_names", [])
        self._feature_importance_dict = data.get("feature_importance", {})
        self._fitted = data.get("fitted", False)

        # 重建骨架并加载权重
        tabnet_params = {
            "n_d": self.params["n_d"],
            "n_a": self.params["n_a"],
            "n_steps": self.params["n_steps"],
            "gamma": self.params["gamma"],
            "n_independent": self.params["n_independent"],
            "n_shared": self.params["n_shared"],
            "momentum": self.params["momentum"],
            "mask_type": self.params["mask_type"],
            "seed": self.params["random_state"],
        }
        self._model = TabNetRegressor(
            input_dim=self.params["input_dim"],
            output_dim=self.params["output_dim"],
            **tabnet_params,
        )

        weight_path = path.replace(".pkl", ".zip")
        if os.path.exists(weight_path) and hasattr(self._model, "load_model"):
            self._model.load_model(weight_path)

        self.logger.info("TabNet 模型已加载", path=path)
