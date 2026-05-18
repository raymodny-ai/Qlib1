"""
表达式引擎单元测试
"""

import pytest
import pandas as pd
import numpy as np
from src.analyzers.expression_engine import (
    ExpressionCompiler,
    ExpressionEngine,
    CompiledExpression,
)

@pytest.fixture
def compiler():
    return ExpressionCompiler()

@pytest.fixture
def engine():
    return ExpressionEngine()

@pytest.fixture
def sample_df():
    np.random.seed(42)
    dates = pd.date_range("2020-01-01", periods=100, freq="B")
    df = pd.DataFrame({
        "close": 100 + np.cumsum(np.random.randn(100) * 2),
        "volume": np.random.randint(100000, 1000000, 100),
        "revenue": np.random.uniform(100, 500, 100),
        "net_income": np.random.uniform(10, 50, 100),
    }, index=dates)
    return df


class TestExpressionCompiler:
    """表达式编译器测试"""

    def test_compile_simple_constant(self, compiler):
        expr = "1 + 1"
        result = compiler.compile(expr)
        assert isinstance(result, CompiledExpression)
        assert result.source == expr
        assert result.required_fields == set()

    def test_compile_field_reference(self, compiler):
        expr = "$close + $volume"
        result = compiler.compile(expr)
        assert "close" in result.required_fields
        assert "volume" in result.required_fields

    def test_compile_with_builtin_function(self, compiler):
        expr = "Mean($close, 20)"
        result = compiler.compile(expr)
        assert "close" in result.required_fields

    def test_compile_complex_expression(self, compiler):
        expr = "$close / Ref($close, 5) - 1"
        result = compiler.compile(expr)
        assert "close" in result.required_fields

    def test_compile_math_operators(self, compiler):
        expr = "($close + $volume) * 2 / ($revenue - $net_income)"
        result = compiler.compile(expr)
        assert len(result.required_fields) == 4

    def test_compile_with_comparison(self, compiler):
        expr = "$close > 100"
        result = compiler.compile(expr)
        assert "close" in result.required_fields

    def test_compile_with_logic(self, compiler):
        expr = "$close > 100 and $volume > 500000"
        result = compiler.compile(expr)
        assert "close" in result.required_fields
        assert "volume" in result.required_fields

    def test_compile_syntax_error_raises(self, compiler):
        with pytest.raises(ValueError):
            compiler.compile("$close +")

    def test_compile_empty_expression(self, compiler):
        expr = "42"
        result = compiler.compile(expr)
        assert result.required_fields == set()
        assert result.required_functions == set()


class TestExpressionEngine:
    """表达式求值引擎测试"""

    def test_evaluate_constant(self, engine, sample_df):
        result = engine.evaluate("42", sample_df)
        assert isinstance(result, pd.Series)
        assert (result == 42).all()

    def test_evaluate_simple_arithmetic(self, engine, sample_df):
        result = engine.evaluate("$close + 1", sample_df)
        expected = sample_df["close"] + 1
        pd.testing.assert_series_equal(result, expected, check_names=False)

    def test_evaluate_subtraction(self, engine, sample_df):
        result = engine.evaluate("$close - $volume", sample_df)
        expected = sample_df["close"] - sample_df["volume"]
        pd.testing.assert_series_equal(result, expected, check_names=False)

    def test_evaluate_multiplication(self, engine, sample_df):
        result = engine.evaluate("$close * 2", sample_df)
        expected = sample_df["close"] * 2
        pd.testing.assert_series_equal(result, expected, check_names=False)

    def test_evaluate_division(self, engine, sample_df):
        result = engine.evaluate("$close / 2", sample_df)
        expected = sample_df["close"] / 2
        pd.testing.assert_series_equal(result, expected, check_names=False)

    def test_evaluate_power(self, engine, sample_df):
        result = engine.evaluate("$close ** 2", sample_df)
        expected = sample_df["close"] ** 2
        pd.testing.assert_series_equal(result, expected, check_names=False)

    def test_evaluate_ref(self, engine, sample_df):
        result = engine.evaluate("Ref($close, 1)", sample_df)
        expected = sample_df["close"].shift(1)
        pd.testing.assert_series_equal(result, expected, check_names=False)

    def test_evaluate_mean(self, engine, sample_df):
        result = engine.evaluate("Mean($close, 5)", sample_df)
        expected = sample_df["close"].rolling(5, min_periods=1).mean()
        pd.testing.assert_series_equal(result, expected, check_names=False)

    def test_evaluate_std(self, engine, sample_df):
        result = engine.evaluate("Std($close, 5)", sample_df)
        expected = sample_df["close"].rolling(5, min_periods=1).std()
        pd.testing.assert_series_equal(result, expected, check_names=False)

    def test_evaluate_max(self, engine, sample_df):
        result = engine.evaluate("Max($close, 5)", sample_df)
        expected = sample_df["close"].rolling(5, min_periods=1).max()
        pd.testing.assert_series_equal(result, expected, check_names=False)

    def test_evaluate_min(self, engine, sample_df):
        result = engine.evaluate("Min($close, 5)", sample_df)
        expected = sample_df["close"].rolling(5, min_periods=1).min()
        pd.testing.assert_series_equal(result, expected, check_names=False)

    def test_evaluate_sum(self, engine, sample_df):
        result = engine.evaluate("Sum($close, 5)", sample_df)
        expected = sample_df["close"].rolling(5, min_periods=1).sum()
        pd.testing.assert_series_equal(result, expected, check_names=False)

    def test_evaluate_delta(self, engine, sample_df):
        result = engine.evaluate("Delta($close, 1)", sample_df)
        expected = sample_df["close"].diff()
        pd.testing.assert_series_equal(result, expected, check_names=False)

    def test_evaluate_pctchange(self, engine, sample_df):
        result = engine.evaluate("PctChange($close, 1)", sample_df)
        expected = sample_df["close"].pct_change()
        pd.testing.assert_series_equal(result, expected, check_names=False)

    def test_evaluate_shift(self, engine, sample_df):
        result = engine.evaluate("Shift($close, -1)", sample_df)
        expected = sample_df["close"].shift(-1)
        pd.testing.assert_series_equal(result, expected, check_names=False)

    def test_evaluate_log(self, engine, sample_df):
        result = engine.evaluate("Log($close)", sample_df)
        expected = np.log(sample_df["close"].replace(0, np.nan))
        pd.testing.assert_series_equal(result, expected, check_names=False)

    def test_evaluate_abs(self, engine, sample_df):
        result = engine.evaluate("Abs($close - $close.mean())", sample_df)
        expected = (sample_df["close"] - sample_df["close"].mean()).abs()
        pd.testing.assert_series_equal(result, expected, check_names=False)

    def test_evaluate_comparison_gt(self, engine, sample_df):
        result = engine.evaluate("$close > 100", sample_df)
        expected = sample_df["close"] > 100
        pd.testing.assert_series_equal(result, expected, check_names=False)

    def test_evaluate_comparison_eq(self, engine, sample_df):
        result = engine.evaluate("$close == $close", sample_df)
        assert result.all()

    def test_evaluate_comparison_ne(self, engine, sample_df):
        result = engine.evaluate("$close != 0", sample_df)
        assert result.all()

    def test_evaluate_ifexp(self, engine, sample_df):
        result = engine.evaluate(
            "$close if $close > 50 else 50",
            sample_df
        )
        expected = sample_df["close"].clip(lower=50)
        pd.testing.assert_series_equal(result, expected, check_names=False)

    def test_evaluate_bool_and(self, engine, sample_df):
        result = engine.evaluate(
            "$close > 50 and $volume > 100000",
            sample_df
        )
        expected = (sample_df["close"] > 50) & (sample_df["volume"] > 100000)
        pd.testing.assert_series_equal(result, expected, check_names=False)

    def test_evaluate_bool_or(self, engine, sample_df):
        result = engine.evaluate(
            "$close > 200 or $volume < 200000",
            sample_df
        )
        expected = (sample_df["close"] > 200) | (sample_df["volume"] < 200000)
        pd.testing.assert_series_equal(result, expected, check_names=False)

    def test_missing_field_raises(self, engine, sample_df):
        with pytest.raises(KeyError):
            engine.evaluate("$nonexistent + 1", sample_df)

    def test_evaluate_batch(self, engine, sample_df):
        expressions = {
            "momentum5": "$close / Ref($close, 5) - 1",
            "volume_ma5": "Mean($volume, 5)",
            "close_std10": "Std($close, 10)",
        }
        result = engine.evaluate_batch(expressions, sample_df)
        assert isinstance(result, pd.DataFrame)
        assert list(result.columns) == list(expressions.keys())
        assert len(result) == len(sample_df)

    def test_evaluate_batch_with_error(self, engine, sample_df):
        expressions = {
            "good": "$close + 1",
            "bad": "$nonexistent + 1",  # 应被捕获
        }
        result = engine.evaluate_batch(expressions, sample_df)
        assert "good" in result.columns
        assert "bad" in result.columns
        assert result["bad"].isna().all()

    def test_scalar_result_becomes_series(self, engine, sample_df):
        result = engine.evaluate("42", sample_df)
        assert len(result) == len(sample_df)

    def test_evaluate_with_extra_context(self, engine, sample_df):
        result = engine.evaluate("$close * factor", sample_df, {"factor": 2.0})
        expected = sample_df["close"] * 2.0
        pd.testing.assert_series_equal(result, expected, check_names=False)

    def test_negative_index_ref(self, engine, sample_df):
        """Ref(x, -1): x 在未来1天的值 = shift(-1)"""
        result = engine.evaluate("Ref($close, -1)", sample_df)
        expected = sample_df["close"].shift(-1)
        pd.testing.assert_series_equal(result, expected, check_names=False)

    def test_csmean_basic(self, engine, sample_df):
        result = engine.evaluate("CsMean($close)", sample_df)
        # CsMean without group_key returns all-same Series
        assert isinstance(result, pd.Series)
        assert len(result) == len(sample_df)

    def test_unary_negation(self, engine, sample_df):
        result = engine.evaluate("-$close", sample_df)
        expected = -sample_df["close"]
        pd.testing.assert_series_equal(result, expected, check_names=False)
