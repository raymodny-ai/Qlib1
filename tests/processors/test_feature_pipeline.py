"""
Feature Pipeline 单元测试 — 特征清洗与标准化

测试覆盖:
- RobusZScoreNorm: fit/transform/fit_transform
- Fillna: 各策略 (zero/forward/median/cross_sectional/constant)
- CSRankNorm: 横截面排名
- Winsorize: 分位数/标准差缩尾
- DropnaLabel: 标签样本剔除
- FeaturePipeline: 链式处理
"""

import numpy as np
import pandas as pd
import pytest

from src.processors.feature_pipeline import (
    CSRankNorm,
    DropnaLabel,
    FeaturePipeline,
    Fillna,
    RobusZScoreNorm,
    Winsorize,
)


# ===== Fixtures =====

@pytest.fixture
def sample_df():
    """标准样本 DataFrame"""
    np.random.seed(42)
    n = 100
    return pd.DataFrame({
        "date": ["2023-01-03"] * 50 + ["2023-01-04"] * 50,
        "instrument": [f"STOCK_{i:03d}" for i in range(50)] * 2,
        "feature_a": np.random.randn(n) * 10 + 100,
        "feature_b": np.random.randn(n) * 5 + 50,
        "feature_c": np.random.randn(n) * 2 + 10,
    })


@pytest.fixture
def df_with_nans(sample_df):
    """包含 NaN 的样本 DataFrame"""
    df = sample_df.copy()
    df.loc[0, "feature_a"] = np.nan
    df.loc[5, "feature_b"] = np.nan
    df.loc[10:15, "feature_c"] = np.nan
    return df


@pytest.fixture
def df_with_outliers():
    """包含极端值的样本 DataFrame"""
    np.random.seed(42)
    n = 100
    df = pd.DataFrame({
        "feature_a": np.random.randn(n) * 10 + 100,
        "feature_b": np.random.randn(n) * 5 + 50,
    })
    # 注入极端值
    df.loc[0, "feature_a"] = 9999.0
    df.loc[1, "feature_b"] = -9999.0
    return df


@pytest.fixture
def df_with_labels():
    """包含标签列的 DataFrame"""
    np.random.seed(42)
    n = 100
    df = pd.DataFrame({
        "feature_a": np.random.randn(n),
        "feature_b": np.random.randn(n),
        "label": np.random.randn(n) * 0.02,
    })
    # 插入 NaN 标签
    df.loc[5, "label"] = np.nan
    df.loc[10, "label"] = np.nan
    return df


# ===== RobusZScoreNorm 测试 =====

class TestRobustZScoreNorm:
    def test_fit(self, sample_df):
        proc = RobusZScoreNorm(columns=["feature_a", "feature_b"])
        proc.fit(sample_df)
        assert proc.is_fitted
        assert "feature_a" in proc._params["medians"]
        assert "feature_b" in proc._params["mads"]

    def test_transform_reduces_scale(self, sample_df):
        proc = RobusZScoreNorm(columns=["feature_a", "feature_b"], clip_range=3.0)
        proc.fit(sample_df)
        result = proc.transform(sample_df)
        # 标准化后均值应接近 0
        assert abs(result["feature_a"].median()) < 1.0
        # 截断生效
        assert result["feature_a"].max() <= 3.0 + 1e-6
        assert result["feature_a"].min() >= -3.0 - 1e-6

    def test_fit_transform(self, sample_df):
        proc = RobusZScoreNorm(columns=["feature_a"])
        result = proc.fit_transform(sample_df)
        assert "feature_a" in result.columns

    def test_clip_range(self, sample_df):
        proc = RobusZScoreNorm(clip_range=1.0)
        result = proc.fit_transform(sample_df)
        assert result["feature_a"].max() <= 1.0 + 1e-6
        assert result["feature_a"].min() >= -1.0 - 1e-6

    def test_no_clip(self, sample_df):
        proc = RobusZScoreNorm(clip_range=None)
        result = proc.fit_transform(sample_df)
        # 无截断时可能有更大的值
        assert result["feature_a"].max() > 1.0  # 可能超过 1

    def test_all_numeric_columns(self, sample_df):
        proc = RobusZScoreNorm()  # 自动选择所有数值列
        result = proc.fit_transform(sample_df)
        # 非数值列应保持不变
        assert "date" in result.columns

    def test_get_params(self, sample_df):
        proc = RobusZScoreNorm(columns=["feature_a"])
        proc.fit(sample_df)
        params = proc.get_params()
        assert "medians" in params
        assert "mads" in params


# ===== Fillna 测试 =====

class TestFillna:
    def test_fillna_zero(self, df_with_nans):
        proc = Fillna(strategy="zero")
        result = proc.fit_transform(df_with_nans)
        assert not result["feature_a"].isna().any()

    def test_fillna_forward(self, df_with_nans):
        df = df_with_nans.sort_values("date").copy()
        proc = Fillna(strategy="forward")
        result = proc.fit_transform(df)
        assert not result["feature_a"].isna().any()

    def test_fillna_median(self, df_with_nans):
        proc = Fillna(strategy="median")
        result = proc.fit_transform(df_with_nans)
        assert not result["feature_a"].isna().any()

    def test_fillna_cross_sectional(self, df_with_nans):
        proc = Fillna(strategy="cross_sectional")
        result = proc.fit_transform(df_with_nans)
        # 同一天内用均值填充
        assert not result["feature_a"].isna().any()

    def test_fillna_constant(self, df_with_nans):
        proc = Fillna(strategy="constant", fill_value=-1.0)
        result = proc.fit_transform(df_with_nans)
        assert not result["feature_a"].isna().any()

    def test_fillna_specific_columns(self, df_with_nans):
        proc = Fillna(columns=["feature_a"], strategy="zero")
        result = proc.fit_transform(df_with_nans)
        assert not result["feature_a"].isna().any()
        # feature_b 的 NaN 仍存在
        assert result["feature_b"].isna().any()


# ===== CSRankNorm 测试 =====

class TestCSRankNorm:
    def test_rank_values_in_range(self, sample_df):
        proc = CSRankNorm(columns=["feature_a", "feature_b"])
        result = proc.fit_transform(sample_df)
        # 排名应在 [0, 1] 之间
        assert result["feature_a"].min() >= 0
        assert result["feature_a"].max() <= 1.0

    def test_rank_cross_sectional(self, sample_df):
        """同一天内排名独立计算"""
        proc = CSRankNorm(columns=["feature_a"], date_col="date")
        result = proc.fit_transform(sample_df)
        # 每天的最大排名应为 1.0
        for date_val in sample_df["date"].unique():
            day_data = result[result["date"] == date_val]
            assert abs(day_data["feature_a"].max() - 1.0) < 0.01

    def test_rank_ascending(self, sample_df):
        proc_asc = CSRankNorm(columns=["feature_a"], ascending=True)
        result_asc = proc_asc.fit_transform(sample_df)
        # 最大值排名最高
        max_idx = sample_df["feature_a"].idxmax()
        assert result_asc.loc[max_idx, "feature_a"] > 0.95


# ===== Winsorize 测试 =====

class TestWinsorize:
    def test_percentile_winsorize(self, df_with_outliers):
        proc = Winsorize(limits=(0.01, 0.99))
        result = proc.fit_transform(df_with_outliers)
        # 极端值被截断
        assert result["feature_a"].max() < 9999.0
        assert result["feature_b"].min() > -9999.0

    def test_sigma_winsorize(self, df_with_outliers):
        proc = Winsorize(limits=(3, 3), method="sigma")
        result = proc.fit_transform(df_with_outliers)
        assert result["feature_a"].max() < 9999.0

    def test_specific_columns(self, df_with_outliers):
        proc = Winsorize(columns=["feature_a"], limits=(0.05, 0.95))
        result = proc.fit_transform(df_with_outliers)
        # feature_a 被截断
        assert result["feature_a"].max() < 9999.0
        # feature_b 极端值仍在
        assert result["feature_b"].min() == -9999.0

    def test_is_fitted(self, df_with_outliers):
        proc = Winsorize()
        assert not proc.is_fitted
        proc.fit(df_with_outliers)
        assert proc.is_fitted


# ===== DropnaLabel 测试 =====

class TestDropnaLabel:
    def test_drop_nan_labels(self, df_with_labels):
        proc = DropnaLabel(label_col="label")
        result = proc.fit_transform(df_with_labels)
        assert len(result) < len(df_with_labels)
        assert not result["label"].isna().any()

    def test_missing_label_col(self, sample_df):
        proc = DropnaLabel(label_col="nonexistent_label")
        result = proc.fit_transform(sample_df)
        assert len(result) == len(sample_df)  # 无变化

    def test_no_nan_labels(self, sample_df):
        sample_df["label"] = 1.0
        proc = DropnaLabel(label_col="label")
        result = proc.fit_transform(sample_df)
        assert len(result) == len(sample_df)


# ===== FeaturePipeline 测试 =====

class TestFeaturePipeline:
    def test_build_from_configs(self):
        configs = [
            {"type": "winsorize", "limits": [0.01, 0.99]},
            {"type": "fillna", "strategy": "zero"},
            {"type": "robust_zscore"},
        ]
        pipeline = FeaturePipeline(configs)
        assert len(pipeline.processors) == 3
        assert pipeline.processor_names == ["Winsorize", "Fillna", "RobusZScoreNorm"]

    def test_fit_transform_chain(self, sample_df):
        configs = [
            {"type": "robust_zscore"},
        ]
        pipeline = FeaturePipeline(configs)
        result = pipeline.fit_transform(sample_df)
        assert isinstance(result, pd.DataFrame)

    def test_fit_then_transform(self, sample_df):
        configs = [
            {"type": "winsorize", "limits": [0.05, 0.95]},
            {"type": "fillna", "strategy": "zero"},
        ]
        pipeline = FeaturePipeline(configs)
        pipeline.fit(sample_df)
        assert pipeline.is_fitted
        result = pipeline.transform(sample_df)
        assert len(result) == len(sample_df)

    def test_full_pipeline(self, sample_df):
        configs = [
            {"type": "winsorize", "limits": [0.01, 0.99]},
            {"type": "fillna", "strategy": "zero"},
            {"type": "robust_zscore", "columns": ["feature_a", "feature_b"]},
            {"type": "cs_rank_norm", "columns": ["feature_a"]},
        ]
        pipeline = FeaturePipeline(configs)
        result = pipeline.fit_transform(sample_df)
        assert result["feature_a"].min() >= 0.0
        assert result["feature_a"].max() <= 1.0

    def test_unknown_processor(self, sample_df):
        configs = [
            {"type": "unknown_type"},
            {"type": "fillna", "strategy": "zero"},
        ]
        pipeline = FeaturePipeline(configs)
        # 只有 fillna 被添加
        assert len(pipeline.processors) == 1

    def test_empty_pipeline(self, sample_df):
        pipeline = FeaturePipeline([])
        assert pipeline.is_fitted
        result = pipeline.transform(sample_df)
        pd.testing.assert_frame_equal(result, sample_df)
