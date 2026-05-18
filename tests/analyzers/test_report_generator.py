"""
绩效报告生成器单元测试
"""

import os
import json
import tempfile
import matplotlib
matplotlib.use("Agg")  # 无头环境，禁用 GUI 后端

import pytest
import pandas as pd
import numpy as np
from src.analyzers.report_generator import (
    BacktestAnalyzer,
    PerformanceReport,
    ReportExporter,
    FullReport,
    ICMetrics,
    ReturnMetrics,
    RiskMetrics,
)


@pytest.fixture
def analyzer():
    return BacktestAnalyzer()


@pytest.fixture
def sample_predictions():
    np.random.seed(42)
    dates = pd.date_range("2020-01-01", periods=100, freq="W")
    instruments = [f"STOCK_{i:03d}" for i in range(30)]
    data = np.random.randn(100, 30) * 0.05
    return pd.DataFrame(data, index=dates, columns=instruments)


@pytest.fixture
def sample_returns():
    np.random.seed(99)
    dates = pd.date_range("2020-01-01", periods=100, freq="W")
    instruments = [f"STOCK_{i:03d}" for i in range(30)]
    data = np.random.randn(100, 30) * 0.02
    return pd.DataFrame(data, index=dates, columns=instruments)


@pytest.fixture
def sample_nav():
    np.random.seed(42)
    dates = pd.date_range("2020-01-01", periods=252, freq="B")
    returns = np.random.randn(252) * 0.01 + 0.0005
    nav = (1 + returns).cumprod()
    return pd.Series(nav, index=dates)


@pytest.fixture
def sample_benchmark_nav():
    np.random.seed(99)
    dates = pd.date_range("2020-01-01", periods=252, freq="B")
    returns = np.random.randn(252) * 0.008 + 0.0003
    nav = (1 + returns).cumprod()
    return pd.Series(nav, index=dates)


class TestBacktestAnalyzer:
    """回测分析器测试"""

    def test_compute_ic_basic(self, analyzer, sample_predictions, sample_returns):
        ic = analyzer.compute_ic(sample_predictions, sample_returns)
        assert isinstance(ic, ICMetrics)
        assert len(ic.ic_series) > 0
        assert -1.0 <= ic.ic_mean <= 1.0

    def test_compute_ic_spearman(self, analyzer, sample_predictions, sample_returns):
        ic = analyzer.compute_ic(sample_predictions, sample_returns, method="spearman")
        assert isinstance(ic, ICMetrics)

    def test_compute_ic_pearson(self, analyzer, sample_predictions, sample_returns):
        ic = analyzer.compute_ic(sample_predictions, sample_returns, method="pearson")
        assert isinstance(ic, ICMetrics)

    def test_compute_ic_empty(self, analyzer):
        empty = pd.DataFrame(index=pd.DatetimeIndex([]))
        ic = analyzer.compute_ic(empty, empty)
        assert ic.ic_mean == 0.0

    def test_compute_ic_rank_ic_present(self, analyzer, sample_predictions, sample_returns):
        ic = analyzer.compute_ic(sample_predictions, sample_returns)
        assert len(ic.rank_ic_series) > 0
        assert -1.0 <= ic.rank_ic_mean <= 1.0

    def test_compute_ic_metrics_in_range(self, analyzer, sample_predictions, sample_returns):
        ic = analyzer.compute_ic(sample_predictions, sample_returns)
        assert 0.0 <= ic.ic_positive_ratio <= 1.0
        assert 0.0 <= ic.ic_significant_ratio <= 1.0

    def test_compute_returns(self, analyzer, sample_nav):
        rm = analyzer.compute_returns(sample_nav)
        assert isinstance(rm, ReturnMetrics)
        assert rm.total_return != 0.0

    def test_compute_returns_with_benchmark(self, analyzer, sample_nav, sample_benchmark_nav):
        rm = analyzer.compute_returns(sample_nav, sample_benchmark_nav)
        assert rm.excess_returns is not None
        assert len(rm.excess_returns) > 0

    def test_compute_returns_empty(self, analyzer):
        nav = pd.Series([1.0])
        rm = analyzer.compute_returns(nav)
        assert rm.total_return == 0.0

    def test_compute_returns_metrics(self, analyzer, sample_nav):
        rm = analyzer.compute_returns(sample_nav)
        assert rm.annualized_volatility >= 0
        assert rm.max_drawdown <= 0
        assert 0.0 <= rm.win_rate <= 1.0

    def test_compute_risk(self, analyzer, sample_nav):
        daily_returns = sample_nav.pct_change().dropna()
        rk = analyzer.compute_risk(daily_returns)
        assert isinstance(rk, RiskMetrics)
        assert rk.var_95 <= 0  # VaR 应为负值
        assert rk.var_99 <= rk.var_95  # 99% VaR 更极端

    def test_compute_risk_empty(self, analyzer):
        rk = analyzer.compute_risk(pd.Series(dtype=float))
        assert rk.var_95 == 0.0

    def test_analyze_full(self, analyzer, sample_predictions, sample_returns, sample_nav):
        report = analyzer.analyze(
            sample_predictions, sample_returns, sample_nav,
            model_name="TestModel"
        )
        assert isinstance(report, FullReport)
        assert report.model_name == "TestModel"
        assert report.ic_metrics is not None
        assert report.return_metrics is not None

    def test_analyze_without_nav(self, analyzer, sample_predictions, sample_returns):
        report = analyzer.analyze(sample_predictions, sample_returns, model_name="ICOnly")
        assert report.ic_metrics is not None
        assert report.return_metrics is None

    def test_full_report_to_dict(self, analyzer, sample_predictions, sample_returns, sample_nav):
        report = analyzer.analyze(sample_predictions, sample_returns, sample_nav)
        d = report.to_dict()
        assert "ic_metrics" in d
        assert "return_metrics" in d
        assert "timestamp" in d


class TestPerformanceReport:
    """绩效报告可视化测试"""

    @pytest.fixture
    def full_report(self, analyzer, sample_predictions, sample_returns, sample_nav):
        return analyzer.analyze(sample_predictions, sample_returns, sample_nav, model_name="Test")

    def test_plot_ic_curve(self, full_report):
        pr = PerformanceReport(full_report)
        fig = pr.plot_ic_curve()
        assert fig is not None

    def test_plot_ic_curve_no_data(self):
        pr = PerformanceReport(FullReport())
        fig = pr.plot_ic_curve()
        assert fig is None

    def test_plot_cumulative_return(self, full_report):
        pr = PerformanceReport(full_report)
        fig = pr.plot_cumulative_return()
        assert fig is not None

    def test_plot_cumulative_return_no_data(self):
        pr = PerformanceReport(FullReport())
        fig = pr.plot_cumulative_return()
        assert fig is None

    def test_plot_risk_analysis(self, full_report):
        pr = PerformanceReport(full_report)
        fig = pr.plot_risk_analysis()
        assert fig is not None

    def test_plot_risk_analysis_no_data(self):
        pr = PerformanceReport(FullReport())
        fig = pr.plot_risk_analysis()
        assert fig is None

    def test_to_html(self, full_report):
        pr = PerformanceReport(full_report)
        pr.plot_ic_curve()
        pr.plot_cumulative_return()
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "report.html")
            html = pr.to_html(path)
            assert os.path.exists(path)
            assert "Qlib" in html

    def test_to_json(self, full_report):
        pr = PerformanceReport(full_report)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "report.json")
            pr.to_json(path)
            assert os.path.exists(path)
            with open(path, "r") as f:
                data = json.load(f)
            assert "ic_metrics" in data


class TestReportExporter:
    """报告导出器测试"""

    def test_export_all(self, analyzer, sample_predictions, sample_returns, sample_nav):
        report = analyzer.analyze(sample_predictions, sample_returns, sample_nav)
        exporter = ReportExporter(report)

        with tempfile.TemporaryDirectory() as tmp:
            exporter.export_all(tmp, "test_report")
            # 应生成 HTML 和 JSON
            assert os.path.exists(os.path.join(tmp, "test_report.html"))
            assert os.path.exists(os.path.join(tmp, "test_report.json"))


class TestICMetrics:
    """IC指标 数据类测试"""

    def test_to_dict(self):
        ic = ICMetrics(ic_mean=0.045, icir=0.5)
        d = ic.to_dict()
        assert d["ic_mean"] == 0.045
        assert d["icir"] == 0.5


class TestReturnMetrics:
    """收益指标 数据类测试"""

    def test_to_dict(self):
        rm = ReturnMetrics(total_return=0.15, sharpe_ratio=1.2)
        d = rm.to_dict()
        assert d["total_return"] == 0.15
        assert d["sharpe_ratio"] == 1.2


class TestRiskMetrics:
    """风险指标 数据类测试"""

    def test_to_dict(self):
        rk = RiskMetrics(var_95=-0.02, skewness=-0.3)
        d = rk.to_dict()
        assert d["var_95"] == -0.02
        assert d["skewness"] == -0.3
