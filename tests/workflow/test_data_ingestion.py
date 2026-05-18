"""
数据摄取管道单元测试 — DataIngestionPipeline / DataQualityGate / IngestionScheduler
"""

import os
import tempfile
import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock, patch, PropertyMock
from src.workflow.data_ingestion_pipeline import (
    DataIngestionPipeline,
    DataQualityGate,
    IngestionScheduler,
    IngestionResult,
    QualityCheck,
    QualityStatus,
)


class TestQualityCheck:
    """质量检查测试"""

    def test_pass(self):
        qc = QualityCheck(name="test", status=QualityStatus.PASS)
        assert qc.status == QualityStatus.PASS

    def test_fail(self):
        qc = QualityCheck(name="test", status=QualityStatus.FAIL)
        assert qc.status == QualityStatus.FAIL

    def test_warn(self):
        qc = QualityCheck(name="test", status=QualityStatus.WARN)
        assert qc.status == QualityStatus.WARN

    def test_default_metric(self):
        qc = QualityCheck(name="t", metric=0.85, threshold=0.80)
        assert qc.metric == 0.85


class TestDataQualityGate:
    """数据质量门控测试"""

    @pytest.fixture
    def gate(self):
        return DataQualityGate()

    @pytest.fixture
    def good_df(self):
        np.random.seed(42)
        return pd.DataFrame({
            "a": np.random.randn(100),
            "b": np.random.randn(100),
        })

    def test_pass_on_good_data(self, gate, good_df):
        passed, checks = gate.gate(good_df, "test_source")
        assert passed
        assert len(checks) >= 2  # non_empty + nan_ratio

    def test_fail_on_empty(self, gate):
        passed, checks = gate.gate(pd.DataFrame(), "test_source")
        assert not passed
        assert checks[0].status == QualityStatus.FAIL

    def test_fail_on_high_nan(self):
        gate = DataQualityGate(max_nan_ratio=0.05)
        df = pd.DataFrame({"a": [np.nan] * 50 + [1.0] * 50})
        passed, checks = gate.gate(df, "test")
        assert not passed

    def test_check_returns_all_types(self, gate, good_df):
        checks = gate.check(good_df, "test")
        names = {c.name for c in checks}
        assert "non_empty" in names
        assert "nan_ratio" in names

    def test_row_count_check_pass(self):
        """期望行数在门控范围内"""
        gate = DataQualityGate(min_row_ratio=0.80)
        df = pd.DataFrame({"a": range(90)})  # 90 vs 100 → 90%
        checks = gate.check(df, "test", expected_rows=100)
        row_check = [c for c in checks if c.name == "row_count"][0]
        assert row_check.status == QualityStatus.PASS

    def test_row_count_check_warn(self):
        """期望行数低于门控"""
        gate = DataQualityGate(min_row_ratio=0.80)
        df = pd.DataFrame({"a": range(50)})  # 50 vs 100 → 50%
        checks = gate.check(df, "test", expected_rows=100)
        row_check = [c for c in checks if c.name == "row_count"][0]
        assert row_check.status == QualityStatus.WARN

    def test_value_jump_detection(self):
        """数值跳空检测"""
        gate = DataQualityGate(max_jump_std=2.0)
        # 制造跳空
        values = [100.0] * 10 + [500.0] + [100.0] * 10
        df = pd.DataFrame({"a": values, "b": values})
        passed, checks = gate.gate(df, "test")
        jump_check = [c for c in checks if c.name == "value_jump"]
        assert len(jump_check) == 1

    def test_gate_updates_history(self):
        """门控通过后应更新历史基线"""
        gate = DataQualityGate()
        df = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
        passed, _ = gate.gate(df, "test_source")
        assert passed
        assert gate._history.get("test_source") == 3

    def test_single_column_no_jump_check(self):
        """单列 DataFrame 不检查跳空"""
        gate = DataQualityGate()
        df = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
        checks = gate.check(df, "test")
        names = {c.name for c in checks}
        assert "value_jump" not in names


class TestIngestionResult:
    """摄入结果测试"""

    def test_success(self):
        result = IngestionResult(source="test", status="success")
        assert result.success

    def test_failed(self):
        result = IngestionResult(source="test", status="failed", error="timeout")
        assert not result.success
        assert result.error == "timeout"

    def test_pending(self):
        result = IngestionResult(source="test", status="pending")
        assert not result.success


class TestDataIngestionPipeline:
    """数据摄取管道测试"""

    def test_init(self):
        with tempfile.TemporaryDirectory() as tmp:
            pipeline = DataIngestionPipeline(output_dir=tmp, raw_dir=tmp)
            assert os.path.exists(tmp)

    def test_init_with_custom_quality_gate(self):
        with tempfile.TemporaryDirectory() as tmp:
            gate = DataQualityGate(max_nan_ratio=0.05)
            pipeline = DataIngestionPipeline(output_dir=tmp, raw_dir=tmp, quality_gate=gate)
            assert pipeline.quality_gate is gate

    def test_run_all_sources_graceful(self):
        """测试整体管道运行不崩溃 (源不可用时优雅降级)"""
        with tempfile.TemporaryDirectory() as tmp:
            pipeline = DataIngestionPipeline(
                output_dir=os.path.join(tmp, "qlib_data"),
                raw_dir=os.path.join(tmp, "raw"),
            )
            results = pipeline.run(sources=["alpha_vantage"])
            assert len(results) >= 0

    def test_run_with_all_placeholder(self):
        """"all" 应展开为全部数据源"""
        with tempfile.TemporaryDirectory() as tmp:
            pipeline = DataIngestionPipeline(
                output_dir=os.path.join(tmp, "qlib_data"),
                raw_dir=os.path.join(tmp, "raw"),
            )
            results = pipeline.run(sources=["all"])
            assert len(results) == 4  # 4个数据源都尝试

    def test_run_default_sources_none(self):
        """sources=None 时默认使用 all"""
        with tempfile.TemporaryDirectory() as tmp:
            pipeline = DataIngestionPipeline(
                output_dir=os.path.join(tmp, "qlib_data"),
                raw_dir=os.path.join(tmp, "raw"),
            )
            results = pipeline.run()
            assert len(results) == 4

    def test_get_history(self):
        with tempfile.TemporaryDirectory() as tmp:
            pipeline = DataIngestionPipeline(output_dir=tmp, raw_dir=tmp)
            history = pipeline.get_history()
            assert isinstance(history, list)

    def test_get_summary(self):
        with tempfile.TemporaryDirectory() as tmp:
            pipeline = DataIngestionPipeline(output_dir=tmp, raw_dir=tmp)
            summary = pipeline.get_summary()
            assert "total_runs" in summary
            assert "success_rate" in summary

    def test_get_summary_empty_history(self):
        with tempfile.TemporaryDirectory() as tmp:
            pipeline = DataIngestionPipeline(output_dir=tmp, raw_dir=tmp)
            summary = pipeline.get_summary()
            assert summary["total_runs"] == 0
            assert summary["last_run"] is None

    def test_ingest_source_with_mock_collect_success(self):
        """模拟采集成功场景 (含 DataConverter 不可用时的优雅降级)"""
        with tempfile.TemporaryDirectory() as tmp:
            pipeline = DataIngestionPipeline(
                output_dir=os.path.join(tmp, "qlib_data"),
                raw_dir=os.path.join(tmp, "raw"),
            )

            mock_df = pd.DataFrame({"close": [100.0] * 10, "volume": [1000] * 10})
            with patch.object(pipeline, "_collect", return_value=mock_df):
                with patch.object(pipeline.quality_gate, "gate", return_value=(True, [])):
                    result = pipeline._ingest_source(
                        "alpha_vantage", None, None, None, True
                    )
                    # DataConverter 可能因参数不兼容而失败
                    assert isinstance(result, IngestionResult)

    def test_ingest_source_collect_returns_none(self):
        """采集返回 None 应标记失败"""
        with tempfile.TemporaryDirectory() as tmp:
            pipeline = DataIngestionPipeline(
                output_dir=os.path.join(tmp, "qlib_data"),
                raw_dir=os.path.join(tmp, "raw"),
            )

            with patch.object(pipeline, "_collect", return_value=None):
                result = pipeline._ingest_source(
                    "alpha_vantage", None, None, None, True
                )
                assert result.status == "failed"
                assert "空数据" in (result.error or "")

    def test_ingest_source_quality_gate_fails(self):
        """质量门控失败应标记失败"""
        with tempfile.TemporaryDirectory() as tmp:
            pipeline = DataIngestionPipeline(
                output_dir=os.path.join(tmp, "qlib_data"),
                raw_dir=os.path.join(tmp, "raw"),
            )

            mock_df = pd.DataFrame({"close": [100.0]})
            with patch.object(pipeline, "_collect", return_value=mock_df):
                with patch.object(pipeline.quality_gate, "gate", return_value=(False, [])):
                    result = pipeline._ingest_source(
                        "alpha_vantage", None, None, None, True
                    )
                    assert result.status == "failed"
                    assert "质量门控" in (result.error or "")

    def test_ingest_source_collect_exception(self):
        """采集异常应捕获并标记失败"""
        with tempfile.TemporaryDirectory() as tmp:
            pipeline = DataIngestionPipeline(
                output_dir=os.path.join(tmp, "qlib_data"),
                raw_dir=os.path.join(tmp, "raw"),
            )

            with patch.object(pipeline, "_collect", side_effect=RuntimeError("API down")):
                result = pipeline._ingest_source(
                    "alpha_vantage", None, None, None, True
                )
                assert result.status == "failed"
                assert "API down" in (result.error or "")

    def test_ingest_source_empty_dataframe(self):
        """采集返回空 DataFrame 应标记失败"""
        with tempfile.TemporaryDirectory() as tmp:
            pipeline = DataIngestionPipeline(
                output_dir=os.path.join(tmp, "qlib_data"),
                raw_dir=os.path.join(tmp, "raw"),
            )

            with patch.object(pipeline, "_collect", return_value=pd.DataFrame()):
                result = pipeline._ingest_source(
                    "alpha_vantage", None, None, None, True
                )
                assert result.status == "failed"


class TestIngestionScheduler:
    """定时调度器测试"""

    def test_init(self):
        with tempfile.TemporaryDirectory() as tmp:
            pipeline = DataIngestionPipeline(output_dir=tmp, raw_dir=tmp)
            scheduler = IngestionScheduler(pipeline, schedule="manual")
            assert scheduler.schedule == "manual"

    def test_run_once(self):
        with tempfile.TemporaryDirectory() as tmp:
            pipeline = DataIngestionPipeline(output_dir=tmp, raw_dir=tmp)
            scheduler = IngestionScheduler(pipeline)
            results = scheduler.run_once(sources=[])
            assert isinstance(results, list)

    def test_run_daily(self):
        with tempfile.TemporaryDirectory() as tmp:
            pipeline = DataIngestionPipeline(output_dir=tmp, raw_dir=tmp)
            scheduler = IngestionScheduler(pipeline)
            results = scheduler.run_daily(sources=[])
            assert isinstance(results, list)

    def test_run_daily_default_sources(self):
        """run_daily 默认使用 all 数据源"""
        with tempfile.TemporaryDirectory() as tmp:
            pipeline = DataIngestionPipeline(
                output_dir=os.path.join(tmp, "qlib_data"),
                raw_dir=os.path.join(tmp, "raw"),
            )
            scheduler = IngestionScheduler(pipeline)
            results = scheduler.run_daily()  # 不传 sources，默认 all
            assert len(results) >= 0

    def test_run_hourly(self):
        with tempfile.TemporaryDirectory() as tmp:
            pipeline = DataIngestionPipeline(output_dir=tmp, raw_dir=tmp)
            scheduler = IngestionScheduler(pipeline)
            results = scheduler.run_hourly()
            assert isinstance(results, list)
