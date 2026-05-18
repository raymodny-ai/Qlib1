"""
特征清洗与标准化流水线

基于 Qlib DataHandlerLP 理念实现的可学习特征处理管道，
支持可配置的处理算子链式串联。

核心算子:
- RobustZScoreNorm: 基于中位数和 MAD 的稳健标准化
- Fillna: 横截面行业均值/前向填充
- CSRankNorm: 横截面排名标准化
- DropnaLabel: 剔除无效标签样本
- Winsorize: 极值缩尾处理

设计原则:
- 链式调用，算子可插拔
- 训练/推断模式分离（fit/transform）
- 防止推断期数据泄露
"""

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

from src.utils.logger import get_logger


# ===== 抽象基类 =====

class BaseProcessor(ABC):
    """特征处理器抽象基类"""

    def __init__(self, name: str = ""):
        self.name = name or self.__class__.__name__
        self._fitted = False
        self._params: Dict[str, Any] = {}

    @abstractmethod
    def fit(self, df: pd.DataFrame) -> "BaseProcessor":
        """从训练数据中学习处理参数"""
        ...

    @abstractmethod
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """应用处理参数到数据"""
        ...

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """一次完成 fit + transform"""
        return self.fit(df).transform(df)

    @property
    def is_fitted(self) -> bool:
        return self._fitted

    def get_params(self) -> Dict[str, Any]:
        return self._params.copy()


# ===== 核心算子实现 =====

class RobusZScoreNorm(BaseProcessor):
    """
    稳健 Z-Score 标准化

    使用中位数 (median) 和绝对离差中位数 (MAD) 替代传统
    均值和标准差，对极端值具有高度鲁棒性。

    Formula:
        z = (x - median) / (1.4826 * MAD)

    MAD = median(|x_i - median(x)|)
    1.4826 = 正态分布下 MAD → σ 的缩放因子
    """

    def __init__(
        self,
        columns: Optional[List[str]] = None,
        clip_range: float = 3.0,
        epsilon: float = 1e-12,
    ):
        """
        Args:
            columns: 要标准化的列 (None = 所有数值列)
            clip_range: 截断范围 (±N 标准差)
            epsilon: 防止除零的小常数
        """
        super().__init__()
        self.columns = columns
        self.clip_range = clip_range
        self.epsilon = epsilon

    def fit(self, df: pd.DataFrame) -> "RobusZScoreNorm":
        cols = self.columns or self._numeric_columns(df)

        self._params["medians"] = {}
        self._params["mads"] = {}
        self._params["columns"] = cols

        for col in cols:
            if col not in df.columns:
                continue
            values = df[col].dropna()
            if len(values) == 0:
                self._params["medians"][col] = 0.0
                self._params["mads"][col] = 1.0
            else:
                median = np.median(values)
                mad = np.median(np.abs(values - median))
                self._params["medians"][col] = float(median)
                self._params["mads"][col] = float(max(mad, self.epsilon))

        self._fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self._fitted:
            raise RuntimeError("RobustZScoreNorm 尚未 fit，请先调用 fit()")

        df = df.copy()
        cols = self._params.get("columns", [])

        for col in cols:
            if col not in df.columns:
                continue
            median = self._params["medians"].get(col, 0)
            mad = self._params["mads"].get(col, 1)
            df[col] = (df[col] - median) / (1.4826 * mad)

        # 截断
        if self.clip_range:
            for col in cols:
                if col in df.columns:
                    df[col] = df[col].clip(-self.clip_range, self.clip_range)

        return df

    @staticmethod
    def _numeric_columns(df: pd.DataFrame) -> List[str]:
        return [c for c in df.columns if df[c].dtype in ("float64", "float32", "int64", "int32")]


class Fillna(BaseProcessor):
    """
    缺失值填充

    策略:
    - 'cross_sectional': 横截面均值（同一日期所有股票的均值）
    - 'forward': 前向填充（沿时间轴填充）
    - 'zero': 零填充
    - 'median': 中位数填充
    - 自定义常量
    """

    def __init__(
        self,
        columns: Optional[List[str]] = None,
        strategy: str = "cross_sectional",
        fill_value: float = 0.0,
        group_col: Optional[str] = None,  # 行业分组列（用于行业均值填充）
    ):
        super().__init__()
        self.columns = columns
        self.strategy = strategy
        self.fill_value = fill_value
        self.group_col = group_col

    def fit(self, df: pd.DataFrame) -> "Fillna":
        self._params["columns"] = self.columns or self._numeric_columns(df)
        self._fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        cols = self._params.get("columns", [])

        for col in cols:
            if col not in df.columns:
                continue

            if self.strategy == "zero":
                df[col] = df[col].fillna(0.0)

            elif self.strategy == "forward":
                if "date" in df.columns and "instrument" in df.columns:
                    df = df.sort_values(["instrument", "date"])
                    df[col] = df.groupby("instrument")[col].ffill()
                else:
                    df[col] = df[col].ffill()

            elif self.strategy == "median":
                median_val = df[col].median()
                df[col] = df[col].fillna(median_val if not pd.isna(median_val) else 0.0)

            elif self.strategy == "cross_sectional":
                if "date" in df.columns:
                    df[col] = df.groupby("date")[col].transform(
                        lambda x: x.fillna(x.mean())
                    )
                else:
                    df[col] = df[col].fillna(df[col].mean())

            elif self.strategy == "constant":
                df[col] = df[col].fillna(self.fill_value)

            # 二次填充 (横截面均值后可能仍有 NaN)
            df[col] = df[col].fillna(0.0)

        return df

    @staticmethod
    def _numeric_columns(df: pd.DataFrame) -> List[str]:
        return [c for c in df.columns if df[c].dtype in ("float64", "float32", "int64", "int32")]


class CSRankNorm(BaseProcessor):
    """
    横截面排名标准化

    将每个截面的绝对值转化为相对排名（0~1），
    消除市场整体估值水位波动带来的系统性偏差。

    Formula:
        rank_i = rank(x_i) / N  (升序排名 → [0, 1])
    """

    def __init__(
        self,
        columns: Optional[List[str]] = None,
        ascending: bool = True,
        date_col: str = "date",
    ):
        """
        Args:
            columns: 要排名的列
            ascending: True = 值越小排名越低
            date_col: 日期列名
        """
        super().__init__()
        self.columns = columns
        self.ascending = ascending
        self.date_col = date_col

    def fit(self, df: pd.DataFrame) -> "CSRankNorm":
        self._params["columns"] = self.columns or self._numeric_columns(df)
        self._fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        cols = self._params.get("columns", [])

        if self.date_col in df.columns:
            for col in cols:
                if col not in df.columns:
                    continue
                df[col] = df.groupby(self.date_col)[col].rank(
                    ascending=self.ascending, pct=True
                )
        else:
            for col in cols:
                if col not in df.columns:
                    continue
                df[col] = df[col].rank(ascending=self.ascending, pct=True)

        return df

    @staticmethod
    def _numeric_columns(df: pd.DataFrame) -> List[str]:
        return [c for c in df.columns if df[c].dtype in ("float64", "float32", "int64", "int32")]


class Winsorize(BaseProcessor):
    """
    极值缩尾 (Winsorization)

    将超出分位数范围的数值替换为分位数值，
    保留分布形状的同时消除极端值影响。
    """

    def __init__(
        self,
        columns: Optional[List[str]] = None,
        limits: Tuple[float, float] = (0.01, 0.99),
        method: str = "percentile",
    ):
        """
        Args:
            columns: 要缩尾的列
            limits: (下分位, 上分位) 如 (0.01, 0.99)
            method: 'percentile' | 'sigma' (标准差法)
        """
        super().__init__()
        self.columns = columns
        self.limits = limits
        self.method = method

    def fit(self, df: pd.DataFrame) -> "Winsorize":
        cols = self.columns or self._numeric_columns(df)

        self._params["columns"] = cols
        self._params["lower"] = {}
        self._params["upper"] = {}

        for col in cols:
            if col not in df.columns:
                continue
            values = df[col].dropna()

            if self.method == "sigma":
                mean = values.mean()
                std = values.std()
                self._params["lower"][col] = mean - self.limits[0] * std
                self._params["upper"][col] = mean + self.limits[1] * std
            else:
                self._params["lower"][col] = float(values.quantile(self.limits[0]))
                self._params["upper"][col] = float(values.quantile(self.limits[1]))

        self._fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self._fitted:
            raise RuntimeError("Winsorize 尚未 fit")

        df = df.copy()
        cols = self._params.get("columns", [])

        for col in cols:
            if col not in df.columns:
                continue
            lower = self._params["lower"].get(col)
            upper = self._params["upper"].get(col)
            if lower is not None and upper is not None:
                df[col] = df[col].clip(lower, upper)

        return df

    @staticmethod
    def _numeric_columns(df: pd.DataFrame) -> List[str]:
        return [c for c in df.columns if df[c].dtype in ("float64", "float32", "int64", "int32")]


class DropnaLabel(BaseProcessor):
    """
    剔除无效标签样本

    移除缺失收益标签 (label) 的行，确保模型训练的标签完整性。
    在 Qlib 中通常用于过滤停牌/退市期间的无收益样本。
    """

    def __init__(self, label_col: str = "label"):
        """
        Args:
            label_col: 标签列名
        """
        super().__init__()
        self.label_col = label_col

    def fit(self, df: pd.DataFrame) -> "DropnaLabel":
        """DropnaLabel 不需要 fit，仅记录配置"""
        self._params["label_col"] = self.label_col
        self._fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.label_col not in df.columns:
            return df
        before = len(df)
        df = df.dropna(subset=[self.label_col])
        after = len(df)
        logger = get_logger()
        logger.debug("DropnaLabel", dropped=before - after, remaining=after)
        return df


# ===== 特征流水线 =====

@dataclass
class PipelineConfig:
    """特征处理流水线配置"""
    processors: List[Dict[str, Any]] = field(default_factory=list)
    verbose: bool = True


class FeaturePipeline:
    """
    特征处理流水线

    将多个处理器串联为有序管道，对原始特征矩阵执行
    渐进式清洗、标准化和预处理。

    使用示例:
        pipeline = FeaturePipeline([
            {"type": "winsorize", "limits": (0.01, 0.99)},
            {"type": "fillna", "strategy": "cross_sectional"},
            {"type": "robust_zscore"},
            {"type": "dropna_label", "label_col": "label"},
        ])
        pipeline.fit(train_df)
        clean_train = pipeline.transform(train_df)
        clean_test = pipeline.transform(test_df)  # 使用训练集参数
    """

    PROCESSOR_REGISTRY = {
        "robust_zscore": RobusZScoreNorm,
        "robustzscorenorm": RobusZScoreNorm,
        "fillna": Fillna,
        "csranknorm": CSRankNorm,
        "cs_rank_norm": CSRankNorm,
        "winsorize": Winsorize,
        "dropna_label": DropnaLabel,
        "dropnalabel": DropnaLabel,
    }

    def __init__(self, processor_configs: List[Dict[str, Any]]):
        self.logger = get_logger()
        self.processors: List[BaseProcessor] = []
        self._build_processors(processor_configs)

    def _build_processors(self, configs: List[Dict[str, Any]]):
        """根据配置构建处理器链"""
        for cfg in configs:
            proc_type = cfg.pop("type", "").lower()
            cls = self.PROCESSOR_REGISTRY.get(proc_type)
            if cls is None:
                self.logger.warning("未知处理器类型", type=proc_type)
                continue
            self.processors.append(cls(**cfg))

    def fit(self, df: pd.DataFrame) -> "FeaturePipeline":
        """按顺序对所有处理器执行 fit"""
        for proc in self.processors:
            start = time.time()
            proc.fit(df)
            elapsed = round((time.time() - start) * 1000, 1)
            self.logger.debug(
                f"Fitted {proc.name}", elapsed_ms=elapsed, params=proc.get_params()
            )
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """按顺序应用所有处理器的 transform"""
        result = df.copy()
        for proc in self.processors:
            start = time.time()
            result = proc.transform(result)
            elapsed = round((time.time() - start) * 1000, 1)
            self.logger.debug(
                f"Transformed {proc.name}", elapsed_ms=elapsed, shape=result.shape
            )
        return result

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """一次完成 fit + transform"""
        return self.fit(df).transform(df)

    @property
    def processor_names(self) -> List[str]:
        return [p.name for p in self.processors]

    @property
    def is_fitted(self) -> bool:
        return all(p.is_fitted for p in self.processors) if self.processors else True
