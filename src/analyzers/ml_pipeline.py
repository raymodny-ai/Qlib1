"""
ML 模型训练管道 (Machine Learning Pipeline)

Qlib 风格的多模型训练、验证和预测框架。
支持梯度提升树 (LightGBM/XGBoost) 以及深度表格模型。

核心组件:
- BaseForecastModel: 预测模型抽象基类
- LightGBMModel: LightGBM 回归/分类/排序
- XGBoostModel: XGBoost 回归/分类
- MLPipeline: 训练→验证→预测 完整管道
- TimeSeriesSplit: 时间序列交叉验证

设计原则:
- fit/predict 统一接口
- 严格时间序列划分 (防未来数据泄露)
- 模型持久化与可复现性
- 支持早停与自定义损失函数

使用示例:
    from src.analyzers.ml_pipeline import LightGBMModel, MLPipeline
    
    model = LightGBMModel(
        objective="regression",
        num_leaves=64,
        learning_rate=0.05,
    )
    
    pipeline = MLPipeline(model)
    pipeline.fit(X_train, y_train, X_valid, y_valid)
    predictions = pipeline.predict(X_test)
"""

import os
import pickle
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from src.utils.logger import get_logger


# ========================================================================
#  数据结构
# ========================================================================

@dataclass
class TrainingResult:
    """模型训练结果"""
    model_name: str
    train_loss: List[float] = field(default_factory=list)
    valid_loss: List[float] = field(default_factory=list)
    best_iteration: int = 0
    best_score: float = float("inf")
    feature_importance: Dict[str, float] = field(default_factory=dict)
    train_time_ms: float = 0.0
    n_features: int = 0
    n_samples: int = 0

    @property
    def best_valid_loss(self) -> float:
        return min(self.valid_loss) if self.valid_loss else float("inf")


@dataclass
class PredictionResult:
    """模型预测结果"""
    predictions: np.ndarray
    scores: Optional[np.ndarray] = None  # 分类概率
    index: Optional[pd.Index] = None


# ========================================================================
#  抽象基类
# ========================================================================

class BaseForecastModel(ABC):
    """
    预测模型抽象基类

    所有 ML 模型必须实现 fit / predict 接口。
    子类自动注册到模型工厂。
    """

    MODEL_REGISTRY: Dict[str, type] = {}

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        name = cls.__name__.lower().replace("model", "")
        BaseForecastModel.MODEL_REGISTRY[name] = cls
        # 别名
        BaseForecastModel.MODEL_REGISTRY[cls.__name__] = cls

    def __init__(self, **params):
        self.params = params
        self._model: Any = None
        self._fitted = False
        self._feature_names: List[str] = []
        self.logger = get_logger()
        self._encryptor = None  # 惰性初始化 AES-256 加密器

    def _get_encryptor(self):
        """
        获取 AES-256-GCM 加密器 (PRD 5.1: 模型权重静态存储透明加密)

        仅在环境变量 ENCRYPTION_KEY 已设置时启用。
        子类可用此方法在自定义 save/load 中统一加密。
        """
        if self._encryptor is None:
            import os
            if os.environ.get("ENCRYPTION_KEY"):
                from src.security.security import AES256Encryptor
                self._encryptor = AES256Encryptor(key_env="ENCRYPTION_KEY")
        return self._encryptor

    def _encrypt_bytes(self, data: bytes) -> bytes:
        """加密二进制数据 (若加密器可用)"""
        encryptor = self._get_encryptor()
        if encryptor:
            return encryptor.encrypt(data)
        return data

    def _decrypt_bytes(self, data: bytes) -> bytes:
        """解密二进制数据 (若加密器可用)"""
        encryptor = self._get_encryptor()
        if encryptor:
            return encryptor.decrypt(data)
        return data

    @abstractmethod
    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_valid: Optional[np.ndarray] = None,
        y_valid: Optional[np.ndarray] = None,
        feature_names: Optional[List[str]] = None,
    ) -> TrainingResult:
        """训练模型"""
        ...

    @abstractmethod
    def predict(self, X: np.ndarray) -> PredictionResult:
        """生成预测"""
        ...

    @property
    def is_fitted(self) -> bool:
        return self._fitted

    def get_feature_importance(self) -> Dict[str, float]:
        """获取特征重要性 (子类可覆盖)"""
        return {}

    def save(self, path: str):
        """保存模型到磁盘 (AES-256-GCM 透明加密)"""
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        data = {
            "model": self._model,
            "params": self.params,
            "feature_names": self._feature_names,
        }
        import io
        buf = io.BytesIO()
        pickle.dump(data, buf)
        raw = buf.getvalue()
        encrypted = self._encrypt_bytes(raw)
        with open(path, "wb") as f:
            f.write(encrypted)
        self.logger.info("模型已保存", path=path, encrypted=bool(self._get_encryptor()))

    def load(self, path: str):
        """从磁盘加载模型 (AES-256-GCM 透明解密)"""
        with open(path, "rb") as f:
            encrypted = f.read()
        raw = self._decrypt_bytes(encrypted)
        import io
        buf = io.BytesIO(raw)
        data = pickle.load(buf)
        self._model = data["model"]
        self.params = data.get("params", {})
        self._feature_names = data.get("feature_names", [])
        self._fitted = True
        self.logger.info("模型已加载", path=path, encrypted=bool(self._get_encryptor()))

    @staticmethod
    def create(model_type: str, **params) -> "BaseForecastModel":
        """工厂方法: 按名称创建模型"""
        registry = BaseForecastModel.MODEL_REGISTRY
        key = model_type.lower().replace("_", "").replace("-", "")
        cls = registry.get(key)
        if cls is None:
            # 尝试部分匹配
            for name, model_cls in registry.items():
                if key in name.lower():
                    return model_cls(**params)
            raise ValueError(f"未知模型类型: {model_type}，可用: {list(registry.keys())}")
        return cls(**params)


# ========================================================================
#  LightGBM / XGBoost 模型 (自包含于 src.models)
# ========================================================================

# 从独立模型模块导入，避免代码重复。
# 每个模型文件现在包含完整实现 + Rank IC 自动验证。
from src.models.lightgbm_model import LightGBMModel  # noqa: E402, F811
from src.models.xgboost_model import XGBoostModel  # noqa: E402, F811


# ========================================================================
#  ML 训练管道
# ========================================================================

class MLPipeline:
    """
    ML 训练管道 — 编排训练/验证/预测全流程

    严格遵循时间序列划分，防止未来数据泄露。
    支持自定义评估指标和模型持久化。

    使用示例:
        pipeline = MLPipeline(LightGBMModel(num_leaves=64))
        pipeline.fit(X_train, y_train, X_valid, y_valid)
        preds = pipeline.predict(X_test)
        pipeline.save("models/lgb_v1.pkl")
    """

    def __init__(self, model: BaseForecastModel):
        self.model = model
        self.logger = get_logger()
        self._training_result: Optional[TrainingResult] = None
        self._feature_names: List[str] = []

    def fit(
        self,
        X_train: Union[np.ndarray, pd.DataFrame],
        y_train: Union[np.ndarray, pd.Series],
        X_valid: Optional[Union[np.ndarray, pd.DataFrame]] = None,
        y_valid: Optional[Union[np.ndarray, pd.Series]] = None,
    ) -> TrainingResult:
        """
        训练模型

        Args:
            X_train: 训练特征
            y_train: 训练标签
            X_valid: 验证特征 (可选)
            y_valid: 验证标签 (可选)

        Returns:
            TrainingResult
        """
        # 统一转换为 numpy
        if isinstance(X_train, pd.DataFrame):
            self._feature_names = list(X_train.columns)
            X_train = X_train.values
        if isinstance(y_train, pd.Series):
            y_train = y_train.values
        if isinstance(X_valid, pd.DataFrame):
            X_valid = X_valid.values
        if isinstance(y_valid, pd.Series):
            y_valid = y_valid.values

        # 处理无穷值和缺失值
        X_train = np.nan_to_num(X_train.astype(np.float64), nan=0.0, posinf=1e10, neginf=-1e10)
        y_train = np.nan_to_num(y_train.astype(np.float64), nan=0.0)
        if X_valid is not None:
            X_valid = np.nan_to_num(X_valid.astype(np.float64), nan=0.0, posinf=1e10, neginf=-1e10)
            y_valid = np.nan_to_num(y_valid.astype(np.float64), nan=0.0)

        self._training_result = self.model.fit(
            X_train=X_train,
            y_train=y_train,
            X_valid=X_valid,
            y_valid=y_valid,
            feature_names=self._feature_names,
        )

        return self._training_result

    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> PredictionResult:
        """
        生成预测

        Args:
            X: 特征矩阵

        Returns:
            PredictionResult
        """
        if isinstance(X, pd.DataFrame):
            X = X.values

        X = np.nan_to_num(X.astype(np.float64), nan=0.0, posinf=1e10, neginf=-1e10)
        return self.model.predict(X)

    def evaluate_ic(
        self,
        predictions: np.ndarray,
        returns: np.ndarray,
        method: str = "pearson",
    ) -> Dict[str, float]:
        """
        计算预测得分的信息系数 (IC)

        Args:
            predictions: 模型预测值
            returns: 真实未来收益率
            method: 'pearson' | 'spearman'

        Returns:
            {"IC": float, "Rank_IC": float}
        """
        mask = (~np.isnan(predictions)) & (~np.isnan(returns))
        p = predictions[mask]
        r = returns[mask]

        if len(p) < 5:
            return {"IC": 0.0, "Rank_IC": 0.0}

        if method == "spearman":
            from scipy import stats as scipy_stats
            ic = scipy_stats.spearmanr(p, r)[0]
        else:
            ic = np.corrcoef(p, r)[0, 1]

        # Rank IC (始终用 Spearman)
        from scipy import stats as scipy_stats
        rank_ic = scipy_stats.spearmanr(p, r)[0]

        return {
            "IC": float(ic) if not np.isnan(ic) else 0.0,
            "Rank_IC": float(rank_ic) if not np.isnan(rank_ic) else 0.0,
        }

    def evaluate_icir(
        self,
        predictions_list: List[np.ndarray],
        returns_list: List[np.ndarray],
    ) -> Dict[str, float]:
        """
        计算 ICIR (信息系数信息比率)

        ICIR = Mean(IC) / Std(IC)

        Args:
            predictions_list: 每期预测值列表
            returns_list: 每期真实收益率列表

        Returns:
            {"IC_mean": float, "IC_std": float, "ICIR": float,
             "Rank_IC_mean": float, "Rank_IC_std": float, "Rank_ICIR": float}
        """
        ic_values = []
        rank_ic_values = []

        for preds, rets in zip(predictions_list, returns_list):
            ic_info = self.evaluate_ic(preds, rets)
            ic_values.append(ic_info["IC"])
            rank_ic_values.append(ic_info["Rank_IC"])

        ic_arr = np.array(ic_values)
        rank_ic_arr = np.array(rank_ic_values)

        ic_mean = np.mean(ic_arr)
        ic_std = np.std(ic_arr, ddof=1)
        rank_ic_mean = np.mean(rank_ic_arr)
        rank_ic_std = np.std(rank_ic_arr, ddof=1)

        return {
            "IC_mean": round(ic_mean, 6),
            "IC_std": round(ic_std, 6),
            "ICIR": round(ic_mean / (ic_std + 1e-12), 4),
            "Rank_IC_mean": round(rank_ic_mean, 6),
            "Rank_IC_std": round(rank_ic_std, 6),
            "Rank_ICIR": round(rank_ic_mean / (rank_ic_std + 1e-12), 4),
        }

    @property
    def is_fitted(self) -> bool:
        return self.model.is_fitted

    @property
    def training_result(self) -> Optional[TrainingResult]:
        return self._training_result

    @property
    def feature_importance(self) -> Dict[str, float]:
        return self.model.get_feature_importance()

    def save(self, path: str):
        """保存完整管道 (模型 + 训练结果)"""
        self.model.save(path)

    def load(self, path: str):
        """加载完整管道"""
        self.model.load(path)

    @classmethod
    def load_pipeline(cls, path: str) -> "MLPipeline":
        """静态加载方法"""
        import pickle
        with open(path, "rb") as f:
            data = pickle.load(f)
        params = data.get("params", {})
        model_type = params.get("boosting_type", "lightgbm")
        if "objective" in params:
            model = LightGBMModel(**params)
        else:
            model = XGBoostModel(**params)
        pipeline = cls(model)
        pipeline.model._model = data["model"]
        pipeline.model._fitted = True
        pipeline.model._feature_names = data.get("feature_names", [])
        return pipeline


# ========================================================================
#  时间序列数据划分
# ========================================================================

class TimeSeriesSplitter:
    """
    时间序列数据划分器

    严格按时间顺序切分训练/验证/测试集，
    杜绝未来函数泄露。

    使用示例:
        splitter = TimeSeriesSplitter(
            train_ratio=0.7,
            valid_ratio=0.15,
            test_ratio=0.15,
        )
        train, valid, test = splitter.split(df, date_col="date")
    """

    def __init__(
        self,
        train_ratio: float = 0.7,
        valid_ratio: float = 0.15,
        test_ratio: float = 0.15,
    ):
        total = train_ratio + valid_ratio + test_ratio
        if abs(total - 1.0) > 0.001:
            raise ValueError(f"比例之和必须为 1.0，当前: {total}")
        self.train_ratio = train_ratio
        self.valid_ratio = valid_ratio
        self.test_ratio = test_ratio
        self._split_indices: Optional[Tuple[int, int]] = None

    def split(
        self,
        df: pd.DataFrame,
        date_col: str = "date",
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        按时间顺序划分数据

        Args:
            df: 完整数据集 (必须按时间排序)
            date_col: 日期列名

        Returns:
            (train_df, valid_df, test_df)
        """
        if date_col in df.columns:
            df = df.sort_values(date_col).reset_index(drop=True)

        n = len(df)
        train_end = int(n * self.train_ratio)
        valid_end = train_end + int(n * self.valid_ratio)

        self._split_indices = (train_end, valid_end)

        train = df.iloc[:train_end].copy()
        valid = df.iloc[train_end:valid_end].copy()
        test = df.iloc[valid_end:].copy()

        self.logger = get_logger()
        self.logger.info(
            "时间序列划分完成",
            train=len(train),
            valid=len(valid),
            test=len(test),
        )

        return train, valid, test

    def split_xy(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> Tuple[
        Tuple[np.ndarray, np.ndarray],
        Tuple[np.ndarray, np.ndarray],
        Tuple[np.ndarray, np.ndarray],
    ]:
        """
        按时间顺序划分特征矩阵和标签

        Returns:
            ((X_train, y_train), (X_valid, y_valid), (X_test, y_test))
        """
        n = len(X)
        train_end = int(n * self.train_ratio)
        valid_end = train_end + int(n * self.valid_ratio)

        self._split_indices = (train_end, valid_end)

        return (
            (X[:train_end], y[:train_end]),
            (X[train_end:valid_end], y[train_end:valid_end]),
            (X[valid_end:], y[valid_end:]),
        )

    @property
    def split_indices(self) -> Optional[Tuple[int, int]]:
        return self._split_indices
