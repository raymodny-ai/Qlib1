"""
ML 模型训练管道单元测试

注: 这些测试不依赖 LightGBM/XGBoost 安装，
     使用 mock 隔离外部依赖。当库可用时自动启
     用集成测试。
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock, patch, PropertyMock

from src.analyzers.ml_pipeline import (
    BaseForecastModel,
    LightGBMModel,
    XGBoostModel,
    MLPipeline,
    TimeSeriesSplitter,
    TrainingResult,
    PredictionResult,
)


# ===== 辅助 Fixtures =====

@pytest.fixture
def sample_data():
    np.random.seed(42)
    n = 300
    X = np.random.randn(n, 10)
    # 线性关系 + 噪声
    y = 0.5 * X[:, 0] - 0.3 * X[:, 1] + 0.1 * X[:, 2] + np.random.randn(n) * 0.1
    return X, y


@pytest.fixture
def sample_df():
    np.random.seed(42)
    dates = pd.date_range("2020-01-01", periods=100, freq="B")
    df = pd.DataFrame({
        "date": dates,
        "feature_1": np.random.randn(100),
        "feature_2": np.random.randn(100),
        "label": np.random.randn(100) * 0.02,
    })
    return df


# ===== BaseForecastModel 测试 =====

class TestBaseForecastModel:

    def test_model_registry_populated(self):
        assert "lightgbm" in BaseForecastModel.MODEL_REGISTRY
        assert "xgboost" in BaseForecastModel.MODEL_REGISTRY

    def test_factory_create_lightgbm(self):
        model = BaseForecastModel.create("lightgbm", num_leaves=32)
        assert isinstance(model, LightGBMModel)
        assert model.params["num_leaves"] == 32

    def test_factory_create_xgboost(self):
        model = BaseForecastModel.create("xgboost", max_depth=8)
        assert isinstance(model, XGBoostModel)
        assert model.params["max_depth"] == 8

    def test_factory_unknown_model(self):
        with pytest.raises(ValueError, match="未知模型类型"):
            BaseForecastModel.create("nonexistent_model")

    def test_factory_case_insensitive(self):
        model = BaseForecastModel.create("LightGBM")
        assert isinstance(model, LightGBMModel)


# ===== LightGBM Mock 测试 =====

class TestLightGBMModel:

    def test_init_default_params(self):
        model = LightGBMModel()
        assert model.params["objective"] == "regression"
        assert model.params["num_leaves"] == 64
        assert model.params["learning_rate"] == 0.05
        assert not model.is_fitted

    def test_init_custom_params(self):
        model = LightGBMModel(num_leaves=128, learning_rate=0.01)
        assert model.params["num_leaves"] == 128
        assert model.params["learning_rate"] == 0.01

    def test_fit_returns_training_result(
        self, sample_data
    ):
        pytest.importorskip("lightgbm")

        X, y = sample_data
        X_train, X_valid = X[:200], X[200:]
        y_train, y_valid = y[:200], y[200:]

        model = LightGBMModel(n_estimators=10, num_leaves=8)
        result = model.fit(X_train, y_train, X_valid, y_valid)

        assert isinstance(result, TrainingResult)
        assert result.model_name == "LightGBM"
        assert result.n_features == 10
        assert result.n_samples == 200
        assert model.is_fitted

    def test_fit_without_validation(
        self, sample_data
    ):
        pytest.importorskip("lightgbm")

        X, y = sample_data

        model = LightGBMModel(n_estimators=10, num_leaves=8)
        result = model.fit(X, y)

        assert isinstance(result, TrainingResult)
        assert model.is_fitted

    def test_predict_before_fit_raises(self, sample_data):
        X, _ = sample_data
        model = LightGBMModel()
        with pytest.raises(RuntimeError, match="尚未训练"):
            model.predict(X)

    def test_get_feature_importance_before_fit(self):
        model = LightGBMModel()
        assert model.get_feature_importance() == {}


# ===== XGBoost Mock 测试 =====

class TestXGBoostModel:

    def test_init_default_params(self):
        model = XGBoostModel()
        assert model.params["objective"] == "reg:squarederror"
        assert model.params["max_depth"] == 6

    def test_fit_returns_training_result(self, sample_data):
        pytest.importorskip("xgboost")

        X, y = sample_data
        X_train, X_valid = X[:200], X[200:]
        y_train, y_valid = y[:200], y[200:]

        model = XGBoostModel(n_estimators=10, max_depth=4)
        result = model.fit(X_train, y_train, X_valid, y_valid)

        assert isinstance(result, TrainingResult)
        assert result.model_name == "XGBoost"
        assert model.is_fitted

    def test_predict_before_fit_raises(self, sample_data):
        X, _ = sample_data
        model = XGBoostModel()
        with pytest.raises(RuntimeError, match="尚未训练"):
            model.predict(X)


# ===== MLPipeline 测试 =====

class TestMLPipeline:

    def test_init_with_model(self):
        model = LightGBMModel(num_leaves=32)
        pipeline = MLPipeline(model)
        assert pipeline.model is model
        assert not pipeline.is_fitted

    def test_fit_with_dataframe(
        self, sample_data
    ):
        pytest.importorskip("lightgbm")

        X_np, y_np = sample_data

        X_df = pd.DataFrame(X_np, columns=[f"feature_{i}" for i in range(10)])
        y_series = pd.Series(y_np, name="label")

        pipeline = MLPipeline(LightGBMModel(n_estimators=10, num_leaves=8))
        result = pipeline.fit(X_df, y_series)

        assert isinstance(result, TrainingResult)

    def test_predict_returns_prediction_result(
        self, sample_data
    ):
        pytest.importorskip("lightgbm")

        X, y = sample_data

        pipeline = MLPipeline(LightGBMModel(n_estimators=10, num_leaves=8))
        pipeline.fit(X[:200], y[:200])

        result = pipeline.predict(X[200:])
        assert isinstance(result, PredictionResult)
        assert len(result.predictions) == 100

    def test_evaluate_ic_basic(self):
        np.random.seed(42)
        preds = np.random.randn(100)
        returns = 0.3 * preds + np.random.randn(100) * 0.1

        pipeline = MLPipeline(LightGBMModel())
        ic_info = pipeline.evaluate_ic(preds, returns, method="pearson")
        assert "IC" in ic_info
        assert "Rank_IC" in ic_info
        assert abs(ic_info["IC"]) <= 1.0
        assert abs(ic_info["Rank_IC"]) <= 1.0

    def test_evaluate_ic_perfect_correlation(self):
        preds = np.arange(100, dtype=float)
        returns = preds.copy()

        pipeline = MLPipeline(LightGBMModel())
        ic_info = pipeline.evaluate_ic(preds, returns)
        assert abs(ic_info["IC"] - 1.0) < 0.01

    def test_evaluate_ic_with_nan(self):
        preds = np.array([1.0, 2.0, np.nan, 4.0, 5.0])
        returns = np.array([1.0, 2.0, 3.0, np.nan, 5.0])

        pipeline = MLPipeline(LightGBMModel())
        ic_info = pipeline.evaluate_ic(preds, returns)
        assert not np.isnan(ic_info["IC"])

    def test_evaluate_icir(self):
        np.random.seed(42)
        n_periods = 20
        preds_list = []
        returns_list = []
        for _ in range(n_periods):
            p = np.random.randn(50)
            r = 0.3 * p + np.random.randn(50) * 0.2
            preds_list.append(p)
            returns_list.append(r)

        pipeline = MLPipeline(LightGBMModel())
        icir = pipeline.evaluate_icir(preds_list, returns_list)
        assert "ICIR" in icir
        assert "Rank_ICIR" in icir
        assert "IC_mean" in icir

    def test_feature_importance_before_training(self):
        pipeline = MLPipeline(LightGBMModel())
        assert pipeline.feature_importance == {}

    def test_training_result_property(self):
        pipeline = MLPipeline(LightGBMModel())
        assert pipeline.training_result is None


# ===== TimeSeriesSplitter 测试 =====

class TestTimeSeriesSplitter:

    def test_split_ratios_sum_one(self):
        with pytest.raises(ValueError, match="比例之和必须"):
            TimeSeriesSplitter(train_ratio=0.5, valid_ratio=0.3, test_ratio=0.3)

    def test_split_dataframe(self, sample_df):
        splitter = TimeSeriesSplitter(0.7, 0.15, 0.15)
        train, valid, test = splitter.split(sample_df)

        assert len(train) > 0
        assert len(valid) > 0
        assert len(test) > 0
        assert len(train) + len(valid) + len(test) == len(sample_df)

    def test_split_preserves_order(self, sample_df):
        splitter = TimeSeriesSplitter(0.7, 0.15, 0.15)
        train, valid, test = splitter.split(sample_df)

        # 训练集最后一行的日期 < 验证集第一行的日期
        if "date" in sample_df.columns:
            train_max = train["date"].max()
            valid_min = valid["date"].min()
            test_min = test["date"].min()
            assert train_max <= valid_min
            assert valid_min <= test_min

    def test_split_indices(self, sample_df):
        splitter = TimeSeriesSplitter(0.7, 0.15, 0.15)
        splitter.split(sample_df)
        assert splitter.split_indices is not None
        assert len(splitter.split_indices) == 2

    def test_split_xy(self, sample_data):
        X, y = sample_data
        splitter = TimeSeriesSplitter(0.7, 0.15, 0.15)
        (X_tr, y_tr), (X_val, y_val), (X_te, y_te) = splitter.split_xy(X, y)

        assert len(X_tr) + len(X_val) + len(X_te) == len(X)
        assert len(y_tr) + len(y_val) + len(y_te) == len(y)
