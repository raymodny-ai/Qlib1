"""
Alpha因子库单元测试
"""

import pytest
import pandas as pd
import numpy as np
from src.analyzers.alpha_factors import AlphaFactorCalculator


@pytest.fixture
def calculator():
    return AlphaFactorCalculator()


@pytest.fixture
def full_df():
    """包含所有基本面字段的测试 DataFrame"""
    np.random.seed(42)
    n = 100
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    df = pd.DataFrame({
        "date": dates,
        "close": 100 + np.cumsum(np.random.randn(n) * 2),
        "volume": np.random.randint(100000, 1000000, n),
        # 基本面
        "working_capital": np.random.uniform(50, 200, n),
        "total_assets": np.random.uniform(500, 2000, n),
        "retained_earnings": np.random.uniform(20, 300, n),
        "ebit": np.random.uniform(30, 150, n),
        "market_cap": np.random.uniform(1000, 5000, n),
        "total_liabilities": np.random.uniform(200, 1000, n),
        "revenue": np.random.uniform(100, 500, n),
        "book_equity": np.random.uniform(300, 800, n),
    }, index=dates)
    return df


@pytest.fixture
def fscore_df():
    """包含 Piotroski F-Score 所需字段的测试 DataFrame"""
    np.random.seed(42)
    n = 100
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    df = pd.DataFrame({
        "date": dates,
        "instrument": "AAPL",
        "roa": np.random.uniform(-0.1, 0.3, n),
        "operating_cf": np.random.uniform(-10, 100, n),
        "net_income": np.random.uniform(-20, 80, n),
        "total_assets": np.random.uniform(500, 2000, n),
        "long_term_debt": 300 + np.cumsum(np.random.randn(n) * 10),
        "current_ratio": np.random.uniform(0.5, 3.0, n),
        "shares_outstanding": np.random.uniform(100, 200, n),
        "gross_margin": np.random.uniform(0.1, 0.6, n),
        "asset_turnover": np.random.uniform(0.2, 1.5, n),
    }, index=dates)
    return df


class TestAltmanZScore:
    """Altman Z-Score 测试"""

    def test_altman_zscore_basic(self, calculator, full_df):
        z = calculator.altman_zscore(full_df)
        assert isinstance(z, pd.Series)
        assert len(z) == len(full_df)
        assert z.name == "altman_zscore"

    def test_altman_zscore_values(self, calculator, full_df):
        z = calculator.altman_zscore(full_df)
        assert not z.isna().all()

    def test_altman_zscore_missing_columns(self, calculator):
        df = pd.DataFrame({"close": [1, 2, 3]})
        with pytest.raises(KeyError, match="Altman Z-Score"):
            calculator.altman_zscore(df)

    def test_altman_zscore_zero_assets(self, calculator, full_df):
        df = full_df.copy()
        df.loc[df.index[0], "total_assets"] = 0
        z = calculator.altman_zscore(df)
        assert pd.isna(z.iloc[0])  # 除零应产生 NaN

    def test_altman_zscore_zero_liabilities(self, calculator, full_df):
        df = full_df.copy()
        df.loc[df.index[0], "total_liabilities"] = 0
        z = calculator.altman_zscore(df)
        assert pd.isna(z.iloc[0])

    def test_altman_zscore_modified(self, calculator, full_df):
        z = calculator.altman_zscore_modified(full_df)
        assert isinstance(z, pd.Series)
        assert z.name == "altman_zscore_modified"

    def test_altman_zscore_modified_values_differ(self, calculator, full_df):
        z_orig = calculator.altman_zscore(full_df)
        z_mod = calculator.altman_zscore_modified(full_df)
        # 修正版应使用不同系数
        assert not (z_orig == z_mod).all()


class TestPiotroskiFScore:
    """Piotroski F-Score 测试"""

    def test_fscore_basic(self, calculator, fscore_df):
        f = calculator.piotroski_fscore(fscore_df)
        assert isinstance(f, pd.Series)
        assert f.name == "piotroski_fscore"

    def test_fscore_range(self, calculator, fscore_df):
        f = calculator.piotroski_fscore(fscore_df)
        assert f.min() >= 0
        assert f.max() <= 9

    def test_fscore_integer_values(self, calculator, fscore_df):
        f = calculator.piotroski_fscore(fscore_df)
        assert (f.dropna() == f.dropna().astype(int)).all()

    def test_fscore_with_positive_roa(self, calculator):
        """全部 ROA > 0 时应有更多得分"""
        np.random.seed(42)
        n = 50
        df_pos = pd.DataFrame({
            "roa": np.random.uniform(0.01, 0.3, n),
            "operating_cf": np.random.uniform(10, 100, n),
            "net_income": np.random.uniform(10, 80, n),
            "total_assets": 1000,
            "long_term_debt": np.random.uniform(200, 500, n),
            "current_ratio": np.random.uniform(1.0, 3.0, n),
            "shares_outstanding": 100,
            "gross_margin": np.random.uniform(0.2, 0.5, n),
            "asset_turnover": np.random.uniform(0.5, 1.5, n),
        })
        f = calculator.piotroski_fscore(df_pos)
        # 所有 ROA > 0 时得分应 ≥ 1
        assert f.min() >= 1

    def test_fscore_missing_columns(self, calculator):
        df = pd.DataFrame({"close": [1, 2, 3]})
        with pytest.raises(KeyError, match="Piotroski F-Score"):
            calculator.piotroski_fscore(df)


class TestMACD:
    """MACD 测试"""

    def test_macd_basic(self, calculator, full_df):
        result = calculator.macd(full_df)
        assert isinstance(result, pd.DataFrame)
        assert "macd" in result.columns
        assert "macd_signal" in result.columns
        assert "macd_histogram" in result.columns

    def test_macd_length(self, calculator, full_df):
        result = calculator.macd(full_df)
        assert len(result) == len(full_df)

    def test_macd_histogram_is_difference(self, calculator, full_df):
        result = calculator.macd(full_df)
        hist_calc = result["macd"] - result["macd_signal"]
        pd.testing.assert_series_equal(
            result["macd_histogram"], hist_calc, check_names=False
        )

    def test_macd_custom_params(self, calculator, full_df):
        result = calculator.macd(full_df, fast=5, slow=20, signal=7)
        assert len(result) == len(full_df)

    def test_macd_missing_price_col(self, calculator):
        df = pd.DataFrame({"something": [1, 2, 3]})
        with pytest.raises(KeyError):
            calculator.macd(df)

    def test_macd_custom_price_col(self, calculator, full_df):
        result = calculator.macd(full_df, price_col="close")
        assert len(result) == len(full_df)


class TestBollingerBands:
    """Bollinger Bands 测试"""

    def test_bb_basic(self, calculator, full_df):
        result = calculator.bollinger_bands(full_df)
        assert isinstance(result, pd.DataFrame)
        assert "bb_middle" in result.columns
        assert "bb_upper" in result.columns
        assert "bb_lower" in result.columns
        assert "bb_pct_b" in result.columns
        assert "bb_bandwidth" in result.columns

    def test_bb_upper_above_lower(self, calculator, full_df):
        result = calculator.bollinger_bands(full_df)
        valid = result[["bb_upper", "bb_lower"]].dropna()
        assert (valid["bb_upper"] >= valid["bb_lower"]).all()

    def test_bb_middle_is_mean(self, calculator, full_df):
        result = calculator.bollinger_bands(full_df)
        expected_middle = full_df["close"].rolling(20, min_periods=1).mean()
        pd.testing.assert_series_equal(
            result["bb_middle"], expected_middle, check_names=False
        )

    def test_bb_custom_window(self, calculator, full_df):
        result = calculator.bollinger_bands(full_df, window=10)
        assert len(result) == len(full_df)

    def test_bb_custom_num_std(self, calculator, full_df):
        result = calculator.bollinger_bands(full_df, num_std=3.0)
        # 3σ 带应比 2σ 带更宽
        result_2 = calculator.bollinger_bands(full_df, num_std=2.0)
        width_3 = result["bb_bandwidth"].dropna()
        width_2 = result_2["bb_bandwidth"].dropna()
        assert (width_3 >= width_2).all()


class TestRSI:
    """RSI 测试"""

    def test_rsi_basic(self, calculator, full_df):
        r = calculator.rsi(full_df)
        assert isinstance(r, pd.Series)
        assert r.name == "rsi"

    def test_rsi_range(self, calculator, full_df):
        r = calculator.rsi(full_df)
        valid = r.dropna()
        assert valid.min() >= 0
        assert valid.max() <= 100

    def test_rsi_custom_window(self, calculator, full_df):
        r = calculator.rsi(full_df, window=7)
        assert len(r) == len(full_df)


class TestOrthogonalize:
    """正交化测试"""

    def test_orthogonalize_regression(self, calculator):
        np.random.seed(42)
        n = 200
        ref = pd.Series(np.random.randn(n), name="ref")
        target = 0.5 * ref + pd.Series(np.random.randn(n) * 0.1, name="target")

        residual = calculator.orthogonalize(target, ref, method="regression")
        assert isinstance(residual, pd.Series)
        assert residual.name == "target_orth"

    def test_orthogonalize_reduces_correlation(self, calculator):
        np.random.seed(42)
        n = 500
        ref = pd.Series(np.random.randn(n))
        target = 0.8 * ref + pd.Series(np.random.randn(n) * 0.2)

        residual = calculator.orthogonalize(target, ref, method="regression")
        orig_corr = abs(target.corr(ref))
        resid_corr = abs(residual.corr(ref))
        # 正交化后相关性应接近 0
        assert resid_corr < orig_corr
        assert resid_corr < 0.1

    def test_orthogonalize_difference(self, calculator):
        np.random.seed(42)
        n = 100
        ref = pd.Series(np.random.randn(n), name="ref")
        target = pd.Series(np.random.randn(n), name="target")

        residual = calculator.orthogonalize(target, ref, method="difference")
        assert isinstance(residual, pd.Series)
        assert len(residual) == n

    def test_orthogonalize_ratio(self, calculator):
        np.random.seed(42)
        n = 100
        ref = pd.Series(np.abs(np.random.randn(n)) + 1)
        target = pd.Series(np.abs(np.random.randn(n)) + 1)

        residual = calculator.orthogonalize(target, ref, method="ratio")
        assert isinstance(residual, pd.Series)
        assert len(residual) == n

    def test_orthogonalize_unknown_method(self, calculator):
        with pytest.raises(ValueError):
            calculator.orthogonalize(
                pd.Series([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0]),
                pd.Series([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0]),
                method="unknown",
            )

    def test_orthogonalize_multi(self, calculator):
        np.random.seed(42)
        n = 200
        refs = pd.DataFrame({
            "ref1": np.random.randn(n),
            "ref2": np.random.randn(n),
        })
        target = 0.3 * refs["ref1"] + 0.5 * refs["ref2"] + pd.Series(np.random.randn(n) * 0.1)

        residual = calculator.orthogonalize_multi(target, refs)
        assert isinstance(residual, pd.Series)
        assert len(residual) == n

    def test_orthogonalize_small_sample(self, calculator):
        """小样本正交化回退"""
        target = pd.Series([1.0, 2.0, 3.0])
        ref = pd.Series([1.0, 2.0, np.nan])
        residual = calculator.orthogonalize(target, ref)
        assert isinstance(residual, pd.Series)


class TestComputeAll:
    """综合计算测试"""

    def test_compute_all_with_full_df(self, calculator, full_df):
        # 补充 F-Score 所需字段
        df = full_df.copy()
        df["roa"] = np.random.uniform(0, 0.2, len(df))
        df["operating_cf"] = np.random.uniform(10, 50, len(df))
        df["net_income"] = df["ebit"] * 0.7
        df["long_term_debt"] = np.random.uniform(100, 500, len(df))
        df["current_ratio"] = np.random.uniform(0.5, 3, len(df))
        df["shares_outstanding"] = np.random.uniform(100, 200, len(df))
        df["gross_margin"] = np.random.uniform(0.1, 0.6, len(df))
        df["asset_turnover"] = np.random.uniform(0.2, 1.5, len(df))

        result = calculator.compute_all(df)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(df)
        # 应有多个因子列
        assert len(result.columns) > 0

    def test_compute_all_minimal_df(self, calculator):
        """最小 DataFrame (仅有 close)"""
        df = pd.DataFrame({
            "close": [100, 101, 102, 103, 104],
        })
        result = calculator.compute_all(df)
        assert isinstance(result, pd.DataFrame)
        # 应有 MACD、BB、RSI
        assert "macd" in result.columns
        assert "bb_middle" in result.columns
        assert "rsi" in result.columns
