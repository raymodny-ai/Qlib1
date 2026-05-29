"""
PIT Processor 单元测试 — Point-in-Time 索引

测试覆盖:
- PITRecord 创建
- PITIndex 添加记录/排序
- PITIndex.build_from_dataframe
- PIT 查询 (query/query_dataframe/query_feature_matrix)
- 修正历史追溯
- 索引保存/加载
- PITValidator 验证
"""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.processors.pit_processor import (
    PITIndex,
    PITQueryResult,
    PITRecord,
    PITValidator,
)


# ===== Fixtures =====

@pytest.fixture
def sample_records():
    """创建示例 PIT 记录（含修正版本）"""
    return [
        PITRecord(
            instrument="AAPL", field="revenue", period="2023-Q4",
            period_end_date="2023-09-30", value=383_285_000_000,
            filing_date="2023-11-03 16:30:00", is_amended=False, version_index=0,
        ),
        PITRecord(
            instrument="AAPL", field="revenue", period="2023-Q4",
            period_end_date="2023-09-30", value=383_500_000_000,
            filing_date="2023-12-15 10:00:00", is_amended=True, version_index=1,
            amendment_chain=["2023-11-03 16:30:00", "2023-12-15 10:00:00"],
        ),
        PITRecord(
            instrument="AAPL", field="net_income", period="2023-Q4",
            period_end_date="2023-09-30", value=96_995_000_000,
            filing_date="2023-11-03 16:30:00",
        ),
        PITRecord(
            instrument="MSFT", field="revenue", period="2023-Q4",
            period_end_date="2023-06-30", value=211_915_000_000,
            filing_date="2023-07-25 14:00:00",
        ),
    ]


@pytest.fixture
def pit_index(sample_records):
    idx = PITIndex()
    idx.add_records(sample_records)
    idx.sort_index()
    return idx


# ===== PITRecord 测试 =====

class TestPITRecord:
    def test_basic_record(self):
        r = PITRecord(
            instrument="AAPL", field="revenue", period="2023-Q4",
            period_end_date="2023-09-30", value=100.0,
            filing_date="2023-11-03 16:30:00",
        )
        assert r.instrument == "AAPL"
        assert r.field == "revenue"
        assert r.value == 100.0
        assert not r.is_amended
        assert r.version_index == 0

    def test_amended_record(self):
        r = PITRecord(
            instrument="AAPL", field="revenue", period="2023-Q4",
            period_end_date="2023-09-30", value=105.0,
            filing_date="2023-12-15 10:00:00",
            is_amended=True, version_index=1,
            amendment_chain=["2023-11-03", "2023-12-15"],
        )
        assert r.is_amended
        assert len(r.amendment_chain) == 2


# ===== PITIndex 测试 =====

class TestPITIndex:
    def test_add_records_count(self, sample_records):
        idx = PITIndex()
        count = idx.add_records(sample_records)
        assert count == 4

    def test_instrument_set(self, pit_index):
        assert "AAPL" in pit_index._instrument_set
        assert "MSFT" in pit_index._instrument_set

    def test_field_set(self, pit_index):
        assert "revenue" in pit_index._field_set
        assert "net_income" in pit_index._field_set

    def test_sort_index_versions(self, pit_index):
        # AAPL/revenue/2023-Q4 应有两条记录, v0 和 v1
        records = pit_index._index["AAPL"]["revenue"]["2023-Q4"]
        assert len(records) == 2
        assert records[0].version_index == 0
        assert records[1].version_index == 1
        assert records[1].is_amended

    def test_query_pit_original(self, pit_index):
        """查询 2023-11-15: 应返回原始版本 (v0)"""
        result = pit_index.query("AAPL", "2023-11-15", fields=["revenue"])
        assert len(result.records) == 1
        assert result.records[0].value == 383_285_000_000
        assert not result.records[0].is_amended

    def test_query_pit_amended(self, pit_index):
        """查询 2024-01-01: 应返回修正版本 (v1)"""
        result = pit_index.query("AAPL", "2024-01-01", fields=["revenue"])
        assert len(result.records) == 1
        assert result.records[0].value == 383_500_000_000
        assert result.records[0].is_amended

    def test_query_before_filing_returns_empty(self, pit_index):
        """查询发布日前的日期应返回空"""
        result = pit_index.query("AAPL", "2023-10-01", fields=["revenue"])
        assert len(result.records) == 0

    def test_query_unknown_instrument(self, pit_index):
        result = pit_index.query("ZZZZ", "2024-01-01")
        assert len(result.records) == 0

    def test_query_dataframe(self, pit_index):
        df = pit_index.query_dataframe("AAPL", "2024-01-01")
        assert isinstance(df, pd.DataFrame)
        assert len(df) >= 1
        assert "field" in df.columns
        assert "value" in df.columns

    def test_query_feature_matrix(self, pit_index):
        df = pit_index.query_feature_matrix(
            ["AAPL", "MSFT"], "2024-01-01",
            fields=["revenue", "net_income"],
        )
        assert isinstance(df, pd.DataFrame)
        assert "AAPL" in df.index

    def test_query_feature_matrix_newest_period_wins(self):
        """T2.2: 验证同一 field 有多个 period 时，取 period_end_date 最新的值

        构造: 同一 instrument(AAPL) 同一 field(revenue) 有两个 period:
        - 2023-Q4 (period_end=2023-09-30, value=100)
        - 2024-Q1 (period_end=2023-12-31, value=200)

        旧 Bug: 用 result.records[0].period_end_date 作为比较基准，
        若第一个遍历到 2023-Q4 则永远输出 100，导致静默数据污染。
        """
        idx = PITIndex()
        idx.add_records([
            PITRecord(
                instrument="AAPL", field="revenue", period="2024-Q1",
                period_end_date="2023-12-31", value=200.0,
                filing_date="2024-02-01 16:30:00",
            ),
            PITRecord(
                instrument="AAPL", field="revenue", period="2023-Q4",
                period_end_date="2023-09-30", value=100.0,
                filing_date="2023-11-03 16:30:00",
            ),
        ])
        idx.sort_index()

        df = idx.query_feature_matrix(
            ["AAPL"], "2024-03-01",
            fields=["revenue"],
        )
        # 应取 period_end_date 最新 (=2023-12-31) 的值，即 200.0
        assert df.loc["AAPL", "revenue"] == 200.0, (
            f"应取最新 period 的值 200.0，实际: {df.loc['AAPL', 'revenue']}"
        )

    def test_query_feature_matrix_newest_period_wins_reversed_order(self):
        """T2.2b: 同上，但记录添加顺序颠倒，验证确定性行为"""
        idx = PITIndex()
        # 这次把旧 period 放前面（与上例顺序相反）
        idx.add_records([
            PITRecord(
                instrument="AAPL", field="revenue", period="2023-Q4",
                period_end_date="2023-09-30", value=100.0,
                filing_date="2023-11-03 16:30:00",
            ),
            PITRecord(
                instrument="AAPL", field="revenue", period="2024-Q1",
                period_end_date="2023-12-31", value=200.0,
                filing_date="2024-02-01 16:30:00",
            ),
        ])
        idx.sort_index()

        df = idx.query_feature_matrix(
            ["AAPL"], "2024-03-01",
            fields=["revenue"],
        )
        # 无论添加顺序如何，都应取最新 period 的值 200.0
        assert df.loc["AAPL", "revenue"] == 200.0, (
            f"应取最新 period 的值 200.0 (与添加顺序无关)，实际: {df.loc['AAPL', 'revenue']}"
        )

    def test_amendment_history(self, pit_index):
        history = pit_index.get_amendment_history("AAPL", "revenue", "2023-Q4")
        assert len(history) == 2

    def test_detect_restatements(self, pit_index):
        """在 2023-11-15 时, 12月的修正版本尚未发布"""
        restatements = pit_index.detect_restatements("AAPL", "2023-11-15")
        assert len(restatements) > 0  # 有 future amendment detected

    def test_summary(self, pit_index):
        summary = pit_index.summary
        assert summary["instruments"] >= 2
        assert summary["total_records"] >= 4
        assert summary["amended_records"] >= 1

    def test_save_and_load(self, pit_index, tmp_path):
        path = str(tmp_path / "pit_index.parquet")
        pit_index.save(path)

        loaded = PITIndex.load(path)
        assert loaded.summary["total_records"] == pit_index.summary["total_records"]

    def test_build_from_dataframe(self):
        df = pd.DataFrame([
            {"instrument": "AAPL", "filing_date": "2023-11-03", "period": "2023-Q4",
             "period_end_date": "2023-09-30", "revenue": 100.0, "net_income": 20.0},
            {"instrument": "MSFT", "filing_date": "2023-07-25", "period": "2023-Q4",
             "period_end_date": "2023-06-30", "revenue": 200.0, "net_income": 40.0},
        ])
        idx = PITIndex()
        count = idx.build_from_dataframe(df, value_fields=["revenue", "net_income"])
        assert count == 4  # 2 instruments * 2 fields


# ===== PITValidator 测试 =====

class TestPITValidator:
    def test_valid_index(self, pit_index):
        validator = PITValidator()
        result = validator.validate(pit_index)
        assert result["is_valid"]
        assert len(result["errors"]) == 0

    def test_missing_core_fields_warning(self):
        idx = PITIndex()
        idx.add_records([
            PITRecord(instrument="AAPL", field="custom_metric", period="Q1",
                      period_end_date="2023-03-31", value=1.0,
                      filing_date="2023-04-15 10:00:00"),
        ])
        validator = PITValidator()
        result = validator.validate(idx)
        assert "AAPL" in result["stats"]["missing_core_fields"]


# ===== 修正版本链表测试 (Gap 3) =====

class TestRestatementLinkedList:
    """验证 PITRecord._next_version 显式字段和版本链表追溯"""

    @pytest.fixture
    def tri_amendment_index(self):
        """构建三次修正 (v0→v1→v2) 的索引"""
        idx = PITIndex()
        idx.add_records([
            PITRecord(
                instrument="AAPL", field="revenue", period="2023-Q4",
                period_end_date="2023-09-30", value=100.0,
                filing_date="2023-11-03 16:30:00",
            ),
            PITRecord(
                instrument="AAPL", field="revenue", period="2023-Q4",
                period_end_date="2023-09-30", value=105.0,
                filing_date="2023-12-15 10:00:00",
            ),
            PITRecord(
                instrument="AAPL", field="revenue", period="2023-Q4",
                period_end_date="2023-09-30", value=110.0,
                filing_date="2024-02-01 08:00:00",
            ),
        ])
        idx.sort_index()
        return idx

    def test_linked_list_chain(self, tri_amendment_index):
        """验证 _next_version 显式字段正确链接

        v0._next_version → v1.filing_date
        v1._next_version → v2.filing_date
        v2._next_version → None (最新版本)
        """
        records = tri_amendment_index._index["AAPL"]["revenue"]["2023-Q4"]
        assert len(records) == 3

        # v0 → v1
        assert records[0].version_index == 0
        assert records[0]._next_version == "2023-12-15 10:00:00"
        assert records[0].get_next_version() == "2023-12-15 10:00:00"
        assert not records[0].is_latest_version()
        assert records[0].get_previous_version() is None  # 原始版本

        # v1 → v2
        assert records[1].version_index == 1
        assert records[1]._next_version == "2024-02-01 08:00:00"
        assert records[1].get_next_version() == "2024-02-01 08:00:00"
        assert records[1].get_previous_version() == "2023-11-03 16:30:00"
        assert not records[1].is_latest_version()

        # v2 → None (最新)
        assert records[2].version_index == 2
        assert records[2]._next_version is None
        assert records[2].get_next_version() is None
        assert records[2].is_latest_version()
        assert records[2].get_previous_version() == "2023-12-15 10:00:00"

    def test_backfill_prevents_future_amendment(self, tri_amendment_index):
        """验证回测时无法看到未来修正版本

        查询 2023-11-15 (在 v1 和 v2 发布之前):
        - 应只返回 v0 (value=100.0)
        - v0 此时应是最新可见版本
        """
        result = tri_amendment_index.query("AAPL", "2023-11-15", fields=["revenue"])
        assert len(result.records) == 1
        assert result.records[0].value == 100.0
        assert result.records[0].version_index == 0
        # 在 2023-11-15 时, v1 尚未发布, v0 即最新
        assert result.records[0].get_next_version() is not None  # v1 在数据中存在但不影响查询

    def test_query_after_all_amendments_returns_latest(self, tri_amendment_index):
        """查询 2024-03-01 (所有修正已发布): 应返回最新版本 v2"""
        result = tri_amendment_index.query("AAPL", "2024-03-01", fields=["revenue"])
        assert len(result.records) == 1
        assert result.records[0].value == 110.0
        assert result.records[0].version_index == 2
        assert result.records[0].is_latest_version()

    def test_multiple_amendments_amendment_chain(self, tri_amendment_index):
        """验证三次以上修正的链式追溯 (amendment_chain 完整性)"""
        records = tri_amendment_index._index["AAPL"]["revenue"]["2023-Q4"]
        # v0: 原始版本, amendment_chain 为空 (无历史)
        assert records[0].amendment_chain == []
        # v1: amendment_chain = [v0, v1]
        assert records[1].amendment_chain == [
            "2023-11-03 16:30:00",
            "2023-12-15 10:00:00",
        ]
        # v2: amendment_chain = [v0, v1, v2]
        assert records[2].amendment_chain == [
            "2023-11-03 16:30:00",
            "2023-12-15 10:00:00",
            "2024-02-01 08:00:00",
        ]

    def test_mid_revision_query_correct_version(self, tri_amendment_index):
        """查询 2023-12-20 (v1 已发布, v2 尚未):
        应返回 v1 (value=105.0)"""
        result = tri_amendment_index.query("AAPL", "2023-12-20", fields=["revenue"])
        assert len(result.records) == 1
        assert result.records[0].value == 105.0
        assert result.records[0].version_index == 1
        # v1 的下一版本是 v2 (尚未发布)
        assert result.records[0].get_next_version() == "2024-02-01 08:00:00"
