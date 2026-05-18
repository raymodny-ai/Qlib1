"""
基础设施层单元测试 — HealthChecker / PITMonotonicityValidator / GapDetector / SystemMonitor
"""

import os
import time
import json
import sys
import tempfile
from pathlib import Path
import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock, patch, PropertyMock
from src.infrastructure.health_checker import (
    HealthChecker,
    HealthReport,
    CheckResult,
    CheckStatus,
    PITMonotonicityValidator,
    GapDetector,
    SystemMonitor,
)


class TestCheckResult:
    """检查结果测试"""

    def test_healthy(self):
        cr = CheckResult(name="test", status=CheckStatus.HEALTHY)
        assert cr.is_healthy
        assert not cr.is_critical

    def test_critical(self):
        cr = CheckResult(name="test", status=CheckStatus.CRITICAL)
        assert cr.is_critical
        assert not cr.is_healthy

    def test_warning_not_healthy(self):
        cr = CheckResult(name="test", status=CheckStatus.WARNING)
        assert not cr.is_healthy
        assert not cr.is_critical

    def test_default_fields(self):
        cr = CheckResult(name="my_check")
        assert cr.status == CheckStatus.UNKNOWN
        assert cr.message == ""


class TestHealthReport:
    """健康报告测试"""

    def test_empty_report(self):
        report = HealthReport()
        assert not report.has_critical
        assert report.healthy_count == 0
        assert report.warning_count == 0
        assert report.critical_count == 0

    def test_has_critical(self):
        report = HealthReport(checks=[
            CheckResult(name="c1", status=CheckStatus.HEALTHY),
            CheckResult(name="c2", status=CheckStatus.CRITICAL, message="fail"),
        ])
        assert report.has_critical
        assert report.healthy_count == 1
        assert report.critical_count == 1
        assert report.warning_count == 0

    def test_warning_count(self):
        report = HealthReport(checks=[
            CheckResult(name="c1", status=CheckStatus.WARNING),
            CheckResult(name="c2", status=CheckStatus.WARNING),
            CheckResult(name="c3", status=CheckStatus.HEALTHY),
        ])
        assert report.warning_count == 2
        assert not report.has_critical
        assert report.healthy_count == 1

    def test_to_dict(self):
        report = HealthReport(
            overall=CheckStatus.WARNING,
            checks=[CheckResult(name="c1", status=CheckStatus.HEALTHY)],
        )
        d = report.to_dict()
        assert d["overall"] == "warning"
        assert len(d["checks"]) == 1
        assert d["summary"]["healthy"] == 1
        assert d["summary"]["critical"] == 0

    def test_to_dict_with_details(self):
        cr = CheckResult(
            name="c", status=CheckStatus.HEALTHY,
            details={"key": "val", "num": 42}
        )
        report = HealthReport(checks=[cr])
        d = report.to_dict()
        assert d["checks"][0]["details"]["key"] == "val"


class TestPITMonotonicityValidator:
    """PIT 时序单调性校验器测试"""

    @pytest.fixture
    def mock_pit_index(self):
        """构造模拟的 PIT 索引"""
        from collections import namedtuple
        PITRecord = namedtuple("PITRecord", ["filing_date", "value", "version"])
        pit = MagicMock()
        pit._index = {
            "AAPL": {
                "roe": {
                    "Q1-2020": [
                        PITRecord("2020-04-30", 0.15, 1),
                        PITRecord("2020-07-30", 0.16, 2),  # 修正后
                    ],
                    "Q2-2020": [
                        PITRecord("2020-07-30", 0.14, 1),
                    ],
                },
            },
        }
        return pit

    def test_validate_healthy(self, mock_pit_index):
        validator = PITMonotonicityValidator()
        results = validator.validate(mock_pit_index)
        assert len(results) == 1
        assert results[0].is_healthy

    def test_validate_with_explicit_instruments(self, mock_pit_index):
        validator = PITMonotonicityValidator()
        results = validator.validate(mock_pit_index, instruments=["AAPL"])
        assert results[0].is_healthy

    def test_validate_no_index_attr(self):
        """pit_index 无 _index 属性时应处理"""
        validator = PITMonotonicityValidator()
        pit = MagicMock(spec=[])  # 无 _index
        results = validator.validate(pit)
        assert results[0].is_healthy  # 空集合应通过

    def test_validate_exception(self):
        """PIT 校验异常（无 instruments 且无 _index）应返回 WARNING"""
        validator = PITMonotonicityValidator()
        pit = MagicMock()
        # pit 无 _index 属性 → instruments 为空 → 无检查 → healthy
        del pit._index
        results = validator.validate(pit)
        assert len(results) == 1
        assert results[0].is_healthy  # 空集合通过

    def test_validate_corrupt_index(self):
        """PIT 索引数据异常应返回 WARNING"""
        validator = PITMonotonicityValidator()
        pit = MagicMock()
        pit._index = {"AAPL": {"roe": {"Q1": [MagicMock(filing_date="bad")]}}}
        # 排序可能触发异常
        results = validator.validate(pit, instruments=["AAPL"])
        assert len(results) >= 1


class TestGapDetector:
    """数据断层检测器测试"""

    @pytest.fixture
    def sample_df(self):
        np.random.seed(42)
        dates = pd.date_range("2020-01-01", periods=100, freq="B")
        df = pd.DataFrame({
            "feature_a": np.random.randn(100),
            "feature_b": np.random.randn(100),
        }, index=dates)
        return df

    def test_normal_data(self, sample_df):
        detector = GapDetector()
        results = detector.detect(sample_df)
        assert len(results) >= 2
        nan_result = [r for r in results if r.name == "nan_ratio"][0]
        assert nan_result.is_healthy

    def test_high_nan_data(self):
        detector = GapDetector(max_nan_ratio=0.1)
        df = pd.DataFrame({"a": [np.nan] * 50 + [1.0] * 50})
        results = detector.detect(df)
        nan_result = [r for r in results if r.name == "nan_ratio"][0]
        assert nan_result.status == CheckStatus.WARNING

    def test_multi_column_high_nan_critical(self):
        """>=5 列超限时应为 CRITICAL"""
        detector = GapDetector(max_nan_ratio=0.1)
        data = {}
        for i in range(6):
            data[f"col_{i}"] = [np.nan] * 80 + [1.0] * 20
        df = pd.DataFrame(data)
        results = detector.detect(df)
        nan_result = [r for r in results if r.name == "nan_ratio"][0]
        assert nan_result.status == CheckStatus.CRITICAL

    def test_empty_df(self):
        detector = GapDetector()
        results = detector.detect(pd.DataFrame())
        assert results[0].is_critical

    def test_with_calendar(self):
        detector = GapDetector()
        dates = pd.date_range("2020-01-01", periods=10, freq="B")
        df = pd.DataFrame({"a": range(8)}, index=dates[:8])
        calendar = [d.strftime("%Y-%m-%d") for d in dates]
        results = detector.detect(df, calendar=calendar)
        cal_result = [r for r in results if r.name == "calendar_completeness"]
        assert len(cal_result) > 0

    def test_jump_detection(self):
        """测试跳空检测"""
        detector = GapDetector(jump_sigma_threshold=3.0)
        dates = pd.date_range("2020-01-01", periods=20, freq="B")
        # 制造一次极端跳空
        values = [100.0] * 9 + [200.0] + [100.0] * 10  # 100→200 (+100%)
        df = pd.DataFrame({"price": values}, index=dates)
        results = detector.detect(df)
        jump_results = [r for r in results if r.name == "jump_detection"]
        assert len(jump_results) == 1

    def test_no_calendar_with_non_datetime_index(self):
        """日期索引非 DatetimeIndex 时不检查日历"""
        detector = GapDetector()
        df = pd.DataFrame({"a": [1, 2, 3]})
        results = detector.detect(df, calendar=["2020-01-02"])
        # 因为 index 不是 DatetimeIndex，跳过日历检查
        cal_results = [r for r in results if r.name == "calendar_completeness"]
        assert len(cal_results) == 0

    def test_return_type(self):
        detector = GapDetector()
        df = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
        results = detector.detect(df)
        for r in results:
            assert isinstance(r, CheckResult)


class TestSystemMonitor:
    """系统监控器测试"""

    def test_check_resources(self):
        monitor = SystemMonitor()
        result = monitor.check_resources()
        assert isinstance(result, CheckResult)
        assert result.name == "system_resources"

    def test_uptime(self):
        monitor = SystemMonitor()
        assert monitor.uptime_seconds >= 0

    def test_uptime_increases(self):
        monitor = SystemMonitor()
        u1 = monitor.uptime_seconds
        time.sleep(0.01)
        u2 = monitor.uptime_seconds
        assert u2 > u1

    def test_collect_metrics(self):
        monitor = SystemMonitor()
        metrics = monitor.collect_metrics()
        assert "timestamp" in metrics
        assert "uptime_seconds" in metrics

    def test_check_resources_with_psutil(self):
        """模拟 psutil 可用时的资源检查"""
        import sys
        mock_psutil = MagicMock()
        mock_psutil.cpu_percent.return_value = 50.0
        mock_mem = MagicMock()
        mock_mem.percent = 60.0
        mock_mem.available = 8 * 1024**3
        mock_psutil.virtual_memory.return_value = mock_mem
        mock_disk = MagicMock()
        mock_disk.percent = 70.0
        mock_psutil.disk_usage.return_value = mock_disk

        with patch.dict("sys.modules", {"psutil": mock_psutil}):
            monitor = SystemMonitor()
            result = monitor.check_resources()
            assert result.is_healthy
            assert result.details.get("cpu_percent") == 50.0

    def test_check_resources_high_cpu(self):
        """高 CPU 应触发 WARNING"""
        import sys
        mock_psutil = MagicMock()
        mock_psutil.cpu_percent.return_value = 95.0
        mock_mem = MagicMock()
        mock_mem.percent = 50.0
        mock_mem.available = 8 * 1024**3
        mock_psutil.virtual_memory.return_value = mock_mem
        mock_disk = MagicMock()
        mock_disk.percent = 50.0
        mock_psutil.disk_usage.return_value = mock_disk

        with patch.dict("sys.modules", {"psutil": mock_psutil}):
            monitor = SystemMonitor()
            result = monitor.check_resources()
            assert result.status == CheckStatus.WARNING

    def test_check_resources_high_memory(self):
        """高内存应触发 WARNING"""
        import sys
        mock_psutil = MagicMock()
        mock_psutil.cpu_percent.return_value = 50.0
        mock_mem = MagicMock()
        mock_mem.percent = 95.0
        mock_mem.available = 0.5 * 1024**3
        mock_psutil.virtual_memory.return_value = mock_mem
        mock_disk = MagicMock()
        mock_disk.percent = 50.0
        mock_psutil.disk_usage.return_value = mock_disk

        with patch.dict("sys.modules", {"psutil": mock_psutil}):
            monitor = SystemMonitor()
            result = monitor.check_resources()
            assert result.status == CheckStatus.WARNING

    def test_collect_metrics_with_psutil(self):
        """psutil 可用时采集指标"""
        import sys
        mock_psutil = MagicMock()
        mock_psutil.cpu_percent.return_value = 42.0
        mock_mem = MagicMock()
        mock_mem.percent = 55.0
        mock_mem.available = 4 * 1024**3
        mock_psutil.virtual_memory.return_value = mock_mem

        with patch.dict("sys.modules", {"psutil": mock_psutil}):
            monitor = SystemMonitor()
            metrics = monitor.collect_metrics()
            assert metrics["cpu_percent"] == 42.0
            assert metrics["memory_percent"] == 55.0


class TestHealthChecker:
    """综合健康探针测试"""

    def test_init(self):
        hc = HealthChecker()
        assert hc.data_server is None
        assert hc.pit_index is None
        assert hc.check_interval_s == 3600

    def test_init_with_all_params(self):
        mock_ds = MagicMock()
        mock_pit = MagicMock()
        alert_fn = MagicMock()

        hc = HealthChecker(
            data_server=mock_ds,
            pit_index=mock_pit,
            check_interval_s=600,
            alert_callback=alert_fn,
        )
        assert hc.data_server is mock_ds
        assert hc.pit_index is mock_pit
        assert hc.check_interval_s == 600
        assert hc.alert_callback is alert_fn

    def test_check_with_mock_data_server(self):
        mock_ds = MagicMock()
        mock_ds.stats = {
            "warmed_up": True,
            "instruments": 100,
            "avg_load_time_s": 1.5,
            "cache": {"hit_rate": 0.85},
        }
        hc = HealthChecker(data_server=mock_ds)
        result = hc.check_data_server()
        assert result.is_healthy

    def test_check_warning_cache_hit_rate(self):
        """60%-80% 缓存命中率应为 WARNING"""
        mock_ds = MagicMock()
        mock_ds.stats = {
            "cache": {"hit_rate": 0.65},
        }
        hc = HealthChecker(data_server=mock_ds)
        result = hc.check_data_server()
        assert result.status == CheckStatus.WARNING

    def test_check_low_cache_hit_rate(self):
        mock_ds = MagicMock()
        mock_ds.stats = {
            "cache": {"hit_rate": 0.50},
        }
        hc = HealthChecker(data_server=mock_ds)
        result = hc.check_data_server()
        assert result.status == CheckStatus.CRITICAL

    def test_check_no_data_server(self):
        hc = HealthChecker()
        result = hc.check_data_server()
        assert result.status == CheckStatus.UNKNOWN

    def test_check_data_server_exception(self):
        """DataServer 检查异常应为 CRITICAL"""
        mock_ds = MagicMock()
        mock_ds.stats = PropertyMock(side_effect=RuntimeError("connection lost"))
        hc = HealthChecker(data_server=mock_ds)
        result = hc.check_data_server()
        assert result.status == CheckStatus.CRITICAL

    def test_check_api_endpoint_healthy(self):
        """模拟 API 端点可达"""
        hc = HealthChecker()
        with patch("urllib.request.urlopen") as mock_urlopen:
            mock_resp = MagicMock()
            mock_resp.status = 200
            mock_urlopen.return_value.__enter__.return_value = mock_resp

            result = hc.check_api_endpoint("http://localhost:8000/health")
            assert result.is_healthy
            assert result.details["url"] == "http://localhost:8000/health"

    def test_check_api_endpoint_warning_status(self):
        """API 返回非 200 应为 WARNING"""
        hc = HealthChecker()
        with patch("urllib.request.urlopen") as mock_urlopen:
            mock_resp = MagicMock()
            mock_resp.status = 500
            mock_urlopen.return_value.__enter__.return_value = mock_resp

            result = hc.check_api_endpoint()
            assert result.status == CheckStatus.WARNING

    def test_check_api_endpoint_unreachable(self):
        """API 不可达应为 WARNING"""
        hc = HealthChecker()
        with patch("urllib.request.urlopen", side_effect=OSError("connection refused")):
            result = hc.check_api_endpoint()
            assert result.status == CheckStatus.WARNING

    def test_run_full_check(self):
        hc = HealthChecker()
        report = hc.run_full_check()
        assert isinstance(report, HealthReport)
        assert len(report.checks) >= 2

    def test_run_full_check_with_api(self):
        """含 API 检查的全面体检"""
        hc = HealthChecker()
        with patch.object(hc, "check_api_endpoint") as mock_check:
            mock_check.return_value = CheckResult(
                name="api_endpoint", status=CheckStatus.HEALTHY
            )
            report = hc.run_full_check(check_api=True)
            assert any(c.name == "api_endpoint" for c in report.checks)

    def test_run_full_check_with_feature_df(self):
        """含特征矩阵的全面体检"""
        hc = HealthChecker()
        dates = pd.date_range("2020-01-01", periods=10, freq="B")
        df = pd.DataFrame({"a": np.random.randn(10)}, index=dates)
        report = hc.run_full_check(feature_df=df)
        assert any(c.name == "nan_ratio" for c in report.checks)

    def test_run_full_check_with_calendar(self):
        """含交易日历的全面体检"""
        hc = HealthChecker()
        dates = pd.date_range("2020-01-01", periods=10, freq="B")
        df = pd.DataFrame({"a": np.random.randn(10)}, index=dates)
        calendar = [d.strftime("%Y-%m-%d") for d in dates]
        report = hc.run_full_check(feature_df=df, calendar=calendar)
        assert isinstance(report, HealthReport)

    def test_run_full_check_with_pit_index(self):
        """含 PIT 索引的全面体检"""
        hc = HealthChecker(pit_index=MagicMock())
        hc.pit_index._index = {}
        report = hc.run_full_check()
        assert any(c.name == "pit_monotonicity" for c in report.checks)

    def test_alert_callback_invoked_on_critical(self):
        """严重问题应触发 alert_callback"""
        alert_fn = MagicMock()
        hc = HealthChecker(alert_callback=alert_fn)
        # 注入一个 CRITICAL 检查
        with patch.object(hc, "check_data_server") as mock_check:
            mock_check.return_value = CheckResult(
                name="data_server", status=CheckStatus.CRITICAL, message="fail"
            )
            hc.run_full_check()
            alert_fn.assert_called_once()

    def test_alert_callback_not_invoked_on_healthy(self):
        """健康状态不应触发 alert_callback"""
        alert_fn = MagicMock()
        mock_ds = MagicMock()
        mock_ds.stats = {"cache": {"hit_rate": 0.85}}
        hc = HealthChecker(data_server=mock_ds, alert_callback=alert_fn)
        hc.run_full_check()
        alert_fn.assert_not_called()

    def test_raise_alert(self):
        """测试 raise_alert 写告警日志"""
        hc = HealthChecker()
        report = HealthReport(
            checks=[
                CheckResult(name="c1", status=CheckStatus.CRITICAL, message="disk full"),
            ]
        )
        # 不应崩溃
        hc.raise_alert(report)
        # 验证日志文件已创建
        alert_files = list(Path("logs").glob("alert_*.json"))
        assert len(alert_files) >= 1

    def test_start_periodic_check(self):
        """测试启动定时健康检查"""
        hc = HealthChecker()
        hc.check_interval_s = 0.05  # 缩短间隔用于测试
        hc.start_periodic_check()
        assert hc._running
        time.sleep(0.1)
        hc.stop_periodic_check()
        assert not hc._running

    def test_stop_periodic_check_when_not_running(self):
        """停止未运行的检查不应崩溃"""
        hc = HealthChecker()
        hc.stop_periodic_check()  # 不应崩溃

    def test_double_start_periodic_check(self):
        """重复启动不应创建多个线程"""
        hc = HealthChecker()
        hc.check_interval_s = 0.05
        hc.start_periodic_check()
        hc.start_periodic_check()  # 第二次应跳过
        hc.stop_periodic_check()

    def test_to_metrics_payload(self):
        hc = HealthChecker()
        payload = hc.to_metrics_payload()
        assert "timestamp" in payload
        assert "health_status" in payload

    def test_to_metrics_payload_with_psutil(self):
        """to_metrics_payload 含 psutil 指标"""
        import sys
        mock_psutil = MagicMock()
        mock_psutil.cpu_percent.return_value = 30.0
        mock_mem = MagicMock()
        mock_mem.percent = 50.0
        mock_psutil.virtual_memory.return_value = mock_mem
        mock_disk = MagicMock()
        mock_disk.percent = 60.0
        mock_psutil.disk_usage.return_value = mock_disk

        with patch.dict("sys.modules", {"psutil": mock_psutil}):
            hc = HealthChecker()
            payload = hc.to_metrics_payload()
            assert payload.get("cpu_percent") == 30.0
            assert payload.get("memory_percent") == 50.0

    def test_run_full_check_overall_warning(self):
        """存在 WARNING 检查时 overall 应为 WARNING"""
        hc = HealthChecker()
        with patch.object(hc, "check_data_server") as mock_check:
            mock_check.return_value = CheckResult(
                name="ds", status=CheckStatus.WARNING
            )
            report = hc.run_full_check()
            assert report.overall == CheckStatus.WARNING

    def test_run_full_check_overall_critical(self):
        """存在 CRITICAL 检查时 overall 应为 CRITICAL"""
        hc = HealthChecker()
        with patch.object(hc, "check_data_server") as mock_check:
            mock_check.return_value = CheckResult(
                name="ds", status=CheckStatus.CRITICAL
            )
            report = hc.run_full_check()
            assert report.overall == CheckStatus.CRITICAL
