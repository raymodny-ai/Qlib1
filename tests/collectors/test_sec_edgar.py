"""
SECEdgarCollector 单元测试（含 Mock HTTP 响应）

测试覆盖:
- 初始化与 User-Agent 配置
- edgartools 可用性检测
- CIK 查询 (ticker → CIK 映射)
- Filing 搜索 (edgartools / HTTP 回退)
- XBRL 解析 (edgartools / HTTP 回退)
- _build_xbrl_financials 标签映射
- PIT 时间线构建与查询
- _resolve_tag 辅助函数
- Mock HTTP 集成 (submissions API, companyfacts API)
"""

import json
import os
from dataclasses import asdict
from unittest.mock import AsyncMock, MagicMock, patch

import pandas as pd
import pytest

from src.collectors.base import BatchFetchResult, CollectorConfig, FetchResult
from src.collectors.sec_edgar import (
    FilingType,
    PITTimelineEntry,
    SECEdgarCollector,
    SECEdgarEndpoint,
    SECFiling,
    XBRLFinancials,
    _resolve_tag,
)


# ===== Mock 响应数据 =====

MOCK_CIK_MAP = {
    "0": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."},
    "1": {"cik_str": 789019, "ticker": "MSFT", "title": "Microsoft Corp"},
    "2": {"cik_str": 1018724, "ticker": "AMZN", "title": "Amazon Com Inc"},
}

MOCK_SUBMISSIONS_AAPL = {
    "filings": {
        "recent": {
            "form": ["10-K", "10-Q", "10-Q", "10-K", "10-Q"],
            "filingDate": [
                "2023-11-03", "2023-08-03", "2023-05-04",
                "2022-11-04", "2022-08-03",
            ],
            "accessionNumber": [
                "0000320193-23-000106", "0000320193-23-000080",
                "0000320193-23-000060", "0000320193-22-000108",
                "0000320193-22-000080",
            ],
            "primaryDocument": [
                "aapl-20230930.htm", "aapl-20230701.htm",
                "aapl-20230401.htm", "aapl-20220924.htm",
                "aapl-20220625.htm",
            ],
        }
    }
}

# 不含 10-K 的提交
MOCK_SUBMISSIONS_NO_10K = {
    "filings": {
        "recent": {
            "form": ["8-K", "8-K", "4", "3", "SC 13G"],
            "filingDate": ["2024-01-15", "2023-12-20", "2023-11-01", "2023-10-15", "2023-09-01"],
            "accessionNumber": ["acc1", "acc2", "acc3", "acc4", "acc5"],
            "primaryDocument": ["doc1", "doc2", "doc3", "doc4", "doc5"],
        }
    }
}

MOCK_COMPANY_FACTS_AAPL = {
    "facts": {
        "us-gaap": {
            "Revenues": {
                "units": {
                    "USD": [
                        {"form": "10-K", "fy": 2023, "fp": "FY", "val": 383285000000},
                    ]
                }
            },
            "NetIncomeLoss": {
                "units": {
                    "USD": [
                        {"form": "10-K", "fy": 2023, "fp": "FY", "val": 96995000000},
                    ]
                }
            },
            "Assets": {
                "units": {
                    "USD": [
                        {"form": "10-K", "fy": 2023, "fp": "FY", "val": 352583000000},
                    ]
                }
            },
            "Liabilities": {
                "units": {
                    "USD": [
                        {"form": "10-K", "fy": 2023, "fp": "FY", "val": 290437000000},
                    ]
                }
            },
            "StockholdersEquity": {
                "units": {
                    "USD": [
                        {"form": "10-K", "fy": 2023, "fp": "FY", "val": 62146000000},
                    ]
                }
            },
            "GrossProfit": {
                "units": {
                    "USD": [
                        {"form": "10-K", "fy": 2023, "fp": "FY", "val": 169148000000},
                    ]
                }
            },
            "EarningsPerShareBasic": {
                "units": {
                    "USDperShare": [
                        {"form": "10-K", "fy": 2023, "fp": "FY", "val": 6.16},
                    ]
                }
            },
        }
    }
}


# ===== Helpers =====

def sample_filing(
    ticker="AAPL", cik="0000320193", ftype="10-K",
    acc="0000320193-23-000106", filing_date="2023-11-03 16:30:00",
    period_end="2023-09-30", url="", amended=False,
) -> SECFiling:
    return SECFiling(
        ticker=ticker, cik=cik, filing_type=ftype,
        accession_number=acc, filing_date=filing_date,
        period_end_date=period_end, file_url=url,
        is_amended=amended,
    )


@pytest.fixture
def collector():
    """创建 SECEdgarCollector 实例"""
    return SECEdgarCollector(user_agent="test-agent@example.com")


# ===== 初始化测试 =====

class TestSECInitialization:
    """SECEdgarCollector 初始化"""

    def test_init_with_user_agent(self):
        c = SECEdgarCollector(user_agent="custom@test.com")
        assert c._user_agent == "custom@test.com"

    def test_init_from_env(self):
        with patch.dict(os.environ, {"SEC_EDGAR_USER_AGENT": "env@test.com"}):
            c = SECEdgarCollector()
            assert c._user_agent == "env@test.com"

    def test_init_default_user_agent(self):
        with patch.dict(os.environ, {}, clear=True):
            c = SECEdgarCollector()
            assert "Qlib-US-Fundamental" in c._user_agent

    def test_rate_limiter_configured(self, collector):
        assert collector.rate_limiter is not None

    def test_pit_index_initially_empty(self, collector):
        assert collector._pit_index == {}

    def test_default_headers(self, collector):
        headers = collector._default_headers()
        assert headers["Host"] == "www.sec.gov"
        assert "User-Agent" in headers

    def test_build_url(self, collector):
        url = collector._build_url("some/path")
        assert url.startswith("https://www.sec.gov/")

    def test_collect_daily_prices_returns_empty(self, collector):
        """SEC EDGAR 不提供量价数据"""
        async def _run():
            batch = await collector.collect_daily_prices(["AAPL"])
            assert batch.total_count == 1
            assert len(batch.results) == 0
        import asyncio
        asyncio.run(_run())


# ===== FilingType 枚举测试 =====

class TestFilingType:
    """FilingType 枚举"""

    def test_10k_value(self):
        assert FilingType.K10.value == "10-K"

    def test_10q_value(self):
        assert FilingType.Q10.value == "10-Q"

    def test_8k_value(self):
        assert FilingType.K8.value == "8-K"

    def test_13f_value(self):
        assert FilingType.F13.value == "13F-HR"

    def test_13fa_value(self):
        assert FilingType.F13A.value == "13F-HR/A"


# ===== XBRLFinancials 构建测试 =====

class TestBuildXBRLFinancials:
    """_build_xbrl_financials 标签映射"""

    def test_build_with_revenue(self, collector):
        filing = sample_filing()
        tags = {"revenues": 100000000}
        xbrl = collector._build_xbrl_financials(filing, tags)
        assert xbrl.revenue == 100000000.0

    def test_build_with_alternative_revenue_tag(self, collector):
        filing = sample_filing()
        tags = {"salesrevenuenet": 200000000}
        xbrl = collector._build_xbrl_financials(filing, tags)
        assert xbrl.revenue == 200000000.0

    def test_build_missing_optional_fields(self, collector):
        filing = sample_filing()
        tags = {}  # 空标签
        xbrl = collector._build_xbrl_financials(filing, tags)
        assert xbrl.revenue is None
        assert xbrl.ticker == "AAPL"
        assert xbrl.cik == "0000320193"
        assert xbrl.accession_number == "0000320193-23-000106"

    def test_build_all_tags(self, collector):
        """完整标签映射覆盖"""
        filing = sample_filing()
        tags = {
            "revenues": 100,
            "netincomeloss": 20,
            "assets": 500,
            "assetscurrent": 150,
            "liabilities": 300,
            "liabilitiescurrent": 100,
            "stockholdersequity": 200,
            "retainedearningsaccumulateddeficit": 50,
            "operatingincomeloss": 30,
            "ebit": 25,
            "grossprofit": 60,
            "netcashprovidedbyusedinoperatingactivities": 40,
            "earningspersharebasic": 5.0,
            "earningspersharediluted": 4.8,
            "commonstocksharesoutstanding": 10000000,
            "longtermdebt": 80,
            "cashandcashequivalentsatcarryingvalue": 30,
        }
        xbrl = collector._build_xbrl_financials(filing, tags)
        assert xbrl.revenue == 100.0
        assert xbrl.net_income == 20.0
        assert xbrl.total_assets == 500.0
        assert xbrl.total_current_assets == 150.0
        assert xbrl.total_liabilities == 300.0
        assert xbrl.current_liabilities == 100.0
        assert xbrl.total_equity == 200.0
        assert xbrl.retained_earnings == 50.0
        assert xbrl.operating_income == 30.0
        assert xbrl.ebit == 25.0
        assert xbrl.gross_profit == 60.0
        assert xbrl.operating_cash_flow == 40.0
        assert xbrl.eps_basic == 5.0
        assert xbrl.eps_diluted == 4.8
        assert xbrl.shares_outstanding == 10000000
        assert xbrl.long_term_debt == 80.0
        assert xbrl.cash_and_equivalents == 30.0

    def test_build_invalid_numeric_values(self, collector):
        """无效数值应返回 None 而非崩溃"""
        filing = sample_filing()
        tags = {"revenues": "N/A", "assets": "NotANumber"}
        xbrl = collector._build_xbrl_financials(filing, tags)
        assert xbrl.revenue is None
        assert xbrl.total_assets is None


# ===== CIK 查询测试 =====

class TestCIKLookup:
    """_get_cik — ticker→CIK 映射"""

    @pytest.mark.slow
    @pytest.mark.asyncio
    async def test_get_cik_success(self, collector):
        mock_resp = MagicMock()
        mock_resp.status = 200
        mock_resp.json = AsyncMock(return_value=MOCK_CIK_MAP)
        mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_resp.__aexit__ = AsyncMock(return_value=None)

        mock_session = MagicMock()
        mock_session.get = MagicMock(return_value=mock_resp)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)

        with patch.object(collector, '_get_session', return_value=mock_session):
            cik = await collector._get_cik("AAPL")
            assert cik == "0000320193"

    @pytest.mark.slow
    @pytest.mark.asyncio
    async def test_get_cik_case_insensitive(self, collector):
        mock_resp = MagicMock()
        mock_resp.status = 200
        mock_resp.json = AsyncMock(return_value=MOCK_CIK_MAP)
        mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_resp.__aexit__ = AsyncMock(return_value=None)

        mock_session = MagicMock()
        mock_session.get = MagicMock(return_value=mock_resp)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)

        with patch.object(collector, '_get_session', return_value=mock_session):
            cik = await collector._get_cik("aapl")
            assert cik == "0000320193"

    @pytest.mark.slow
    @pytest.mark.asyncio
    async def test_get_cik_not_found(self, collector):
        mock_resp = MagicMock()
        mock_resp.status = 200
        mock_resp.json = AsyncMock(return_value=MOCK_CIK_MAP)
        mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_resp.__aexit__ = AsyncMock(return_value=None)

        mock_session = MagicMock()
        mock_session.get = MagicMock(return_value=mock_resp)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)

        with patch.object(collector, '_get_session', return_value=mock_session):
            cik = await collector._get_cik("ZZZZ")
            assert cik == ""

    @pytest.mark.slow
    @pytest.mark.asyncio
    async def test_get_cik_http_error(self, collector):
        mock_resp = MagicMock()
        mock_resp.status = 500
        mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_resp.__aexit__ = AsyncMock(return_value=None)

        mock_session = MagicMock()
        mock_session.get = MagicMock(return_value=mock_resp)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)

        with patch.object(collector, '_get_session', return_value=mock_session):
            cik = await collector._get_cik("AAPL")
            assert cik == ""


# ===== Filing 搜索测试 =====

class TestSearchFilings:
    """search_filings — SEC 报告搜索"""

    @pytest.mark.slow
    @pytest.mark.asyncio
    async def test_search_10k_via_http(self, collector):
        """通过 HTTP 回退方案搜索 10-K"""
        # 先 mock CIK 查询
        mock_cik_resp = MagicMock()
        mock_cik_resp.status = 200
        mock_cik_resp.json = AsyncMock(return_value=MOCK_CIK_MAP)
        mock_cik_resp.__aenter__ = AsyncMock(return_value=mock_cik_resp)
        mock_cik_resp.__aexit__ = AsyncMock(return_value=None)

        # mock submissions API
        mock_sub_resp = MagicMock()
        mock_sub_resp.status = 200
        mock_sub_resp.json = AsyncMock(return_value=MOCK_SUBMISSIONS_AAPL)
        mock_sub_resp.__aenter__ = AsyncMock(return_value=mock_sub_resp)
        mock_sub_resp.__aexit__ = AsyncMock(return_value=None)

        mock_session = MagicMock()
        # 第一次调用: CIK lookup, 第二次: submissions
        mock_session.get = MagicMock(side_effect=[mock_cik_resp, mock_sub_resp])
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)

        with patch.object(collector, '_get_session', return_value=mock_session):
            # 强制 HTTP 回退（禁用 edgartools）
            collector._edgar_tools_available = False
            filings = await collector.search_filings("AAPL", FilingType.K10, years=3, limit=5)
            assert len(filings) >= 1
            # 所有 filings 应为 10-K
            for f in filings:
                assert f.filing_type == "10-K"

    @pytest.mark.slow
    @pytest.mark.asyncio
    async def test_search_no_cik_returns_empty(self, collector):
        with patch.object(collector, '_get_cik', AsyncMock(return_value="")):
            collector._edgar_tools_available = False
            filings = await collector.search_filings("ZZZZ", FilingType.K10)
            assert filings == []

    @pytest.mark.slow
    @pytest.mark.asyncio
    async def test_search_no_matching_filings(self, collector):
        """submissions 中无匹配类型的 filing"""
        mock_cik_resp = MagicMock()
        mock_cik_resp.status = 200
        mock_cik_resp.json = AsyncMock(return_value=MOCK_CIK_MAP)
        mock_cik_resp.__aenter__ = AsyncMock(return_value=mock_cik_resp)
        mock_cik_resp.__aexit__ = AsyncMock(return_value=None)

        mock_sub_resp = MagicMock()
        mock_sub_resp.status = 200
        mock_sub_resp.json = AsyncMock(return_value=MOCK_SUBMISSIONS_NO_10K)
        mock_sub_resp.__aenter__ = AsyncMock(return_value=mock_sub_resp)
        mock_sub_resp.__aexit__ = AsyncMock(return_value=None)

        mock_session = MagicMock()
        mock_session.get = MagicMock(side_effect=[mock_cik_resp, mock_sub_resp])
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)

        with patch.object(collector, '_get_session', return_value=mock_session):
            collector._edgar_tools_available = False
            filings = await collector.search_filings("AAPL", FilingType.K10)
            assert filings == []


# ===== XBRL 解析测试 (HTTP 回退) =====

class TestParseXBRLHttp:
    """_parse_xbrl_http — SEC companyfacts API"""

    @pytest.mark.slow
    @pytest.mark.asyncio
    async def test_parse_xbrl_via_http(self, collector):
        mock_resp = MagicMock()
        mock_resp.status = 200
        mock_resp.json = AsyncMock(return_value=MOCK_COMPANY_FACTS_AAPL)
        mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_resp.__aexit__ = AsyncMock(return_value=None)

        mock_session = MagicMock()
        mock_session.get = MagicMock(return_value=mock_resp)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)

        filing = sample_filing(cik="320193")  # 不含前导零
        with patch.object(collector, '_get_session', return_value=mock_session):
            xbrl = await collector._parse_xbrl_http(filing)
            assert xbrl is not None
            assert xbrl.ticker == "AAPL"
            assert xbrl.revenue is not None
            assert xbrl.net_income is not None
            assert xbrl.total_assets is not None

    @pytest.mark.slow
    @pytest.mark.asyncio
    async def test_parse_xbrl_http_error(self, collector):
        mock_resp = MagicMock()
        mock_resp.status = 500
        mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_resp.__aexit__ = AsyncMock(return_value=None)

        mock_session = MagicMock()
        mock_session.get = MagicMock(return_value=mock_resp)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)

        filing = sample_filing(cik="320193")
        with patch.object(collector, '_get_session', return_value=mock_session):
            xbrl = await collector._parse_xbrl_http(filing)
            assert xbrl is None


# ===== _resolve_tag 辅助函数测试 =====

class TestResolveTag:
    """_resolve_tag — 标签解析"""

    def test_exact_match(self):
        tags = {"revenues": 100.0}
        result = _resolve_tag(["revenues"], lambda k: float(tags[k]), tags)
        assert result == 100.0

    def test_prefix_match(self):
        tags = {"revenues_fy2023_q3": 200.0}
        result = _resolve_tag(["revenues"], lambda k: float(tags[k]), tags)
        assert result == 200.0

    def test_multi_candidate_first_wins(self):
        tags = {"netincomeloss": 50.0}
        result = _resolve_tag(["netincomeloss", "profitloss"], lambda k: float(tags[k]), tags)
        assert result == 50.0

    def test_no_match_returns_none(self):
        tags = {"other_tag": 1.0}
        result = _resolve_tag(["revenues"], lambda k: float(tags[k]) if k in tags else None, tags)
        assert result is None

    def test_empty_candidates(self):
        tags = {"revenues": 1.0}
        result = _resolve_tag([], lambda k: float(tags[k]), tags)
        assert result is None


# ===== PIT 时间线测试 =====

class TestPITTimeline:
    """Point-in-Time 时间线"""

    def test_pit_entry_creation(self):
        entry = PITTimelineEntry(
            ticker="AAPL", cik="0000320193",
            financial_period="2023-Q4", period_end_date="2023-09-30",
            filing_date="2023-11-03 16:30:00",
            is_amended=False, version_index=0,
        )
        assert entry.ticker == "AAPL"
        assert entry.financial_period == "2023-Q4"
        assert not entry.is_amended
        assert entry.version_index == 0

    def test_pit_entry_amended(self):
        entry = PITTimelineEntry(
            ticker="AAPL", cik="0000320193",
            financial_period="2023-Q4", period_end_date="2023-09-30",
            filing_date="2023-12-15 10:00:00",
            is_amended=True, version_index=1,
            amendment_chain=["acc1", "acc2"],
        )
        assert entry.is_amended
        assert len(entry.amendment_chain) == 2

    def test_get_pit_without_building_returns_empty(self, collector):
        snapshot = collector.get_pit_data_at("AAPL", "2024-01-01")
        assert snapshot == {}

    @pytest.mark.slow
    @pytest.mark.asyncio
    async def test_build_pit_timeline_mocked(self, collector):
        """通过 Mock CIK + submissions 构建时间线"""
        mock_cik_resp = MagicMock()
        mock_cik_resp.status = 200
        mock_cik_resp.json = AsyncMock(return_value=MOCK_CIK_MAP)
        mock_cik_resp.__aenter__ = AsyncMock(return_value=mock_cik_resp)
        mock_cik_resp.__aexit__ = AsyncMock(return_value=None)

        mock_sub_resp = MagicMock()
        mock_sub_resp.status = 200
        mock_sub_resp.json = AsyncMock(return_value=MOCK_SUBMISSIONS_AAPL)
        mock_sub_resp.__aenter__ = AsyncMock(return_value=mock_sub_resp)
        mock_sub_resp.__aexit__ = AsyncMock(return_value=None)

        mock_session = MagicMock()
        # 1st: CIK, 2nd: 10-K, 3rd: 10-Q
        mock_session.get = MagicMock(side_effect=[mock_cik_resp, mock_sub_resp, mock_sub_resp])
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)

        with patch.object(collector, '_get_session', return_value=mock_session):
            collector._edgar_tools_available = False
            timeline = await collector.build_pit_timeline("AAPL", years=3)
            assert isinstance(timeline, dict)

    def test_get_pit_data_at_with_timeline(self, collector):
        """手动注入 PIT 索引后查询"""
        collector._pit_index["AAPL"] = {
            "2023-09-30": [
                PITTimelineEntry(
                    ticker="AAPL", cik="0000320193",
                    financial_period="2023-Q4", period_end_date="2023-09-30",
                    filing_date="2023-11-03 16:30:00",
                    is_amended=False, version_index=0,
                ),
                PITTimelineEntry(
                    ticker="AAPL", cik="0000320193",
                    financial_period="2023-Q4", period_end_date="2023-09-30",
                    filing_date="2023-12-15 10:00:00",
                    is_amended=True, version_index=1,
                    amendment_chain=["acc_original", "acc_amended"],
                ),
            ]
        }

        # 查询 2023-11-15: 应返回原始版本 (11-03 已在 11-15 之前，12-15 不在)
        snapshot = collector.get_pit_data_at("AAPL", "2023-11-15")
        assert "2023-09-30" in snapshot
        assert not snapshot["2023-09-30"].is_amended

        # 查询 2024-01-01: 应返回修正版本
        snapshot2 = collector.get_pit_data_at("AAPL", "2024-01-01")
        assert snapshot2["2023-09-30"].is_amended
        assert snapshot2["2023-09-30"].version_index == 1

    def test_pit_index_summary(self, collector):
        collector._pit_index["AAPL"] = {
            "2023-Q4": [
                PITTimelineEntry(
                    ticker="AAPL", cik="C1", financial_period="2023-Q4",
                    period_end_date="2023-09-30", filing_date="2023-11-03",
                ),
                PITTimelineEntry(
                    ticker="AAPL", cik="C1", financial_period="2023-Q4",
                    period_end_date="2023-09-30", filing_date="2023-12-15",
                    is_amended=True,
                ),
            ],
            "2023-Q3": [
                PITTimelineEntry(
                    ticker="AAPL", cik="C1", financial_period="2023-Q3",
                    period_end_date="2023-06-30", filing_date="2023-08-03",
                ),
            ],
        }
        summary = collector.pit_index_summary
        assert "AAPL" in summary
        assert summary["AAPL"]["periods"] == 2
        assert summary["AAPL"]["total_entries"] == 3
        assert summary["AAPL"]["periods_with_amendments"] == 1


# ===== 批量收集测试 =====

class TestBatchCollect:
    """batch_collect_10k_10q"""

    @pytest.mark.slow
    @pytest.mark.asyncio
    async def test_batch_collect_10k_10q(self, collector):
        async def mock_search(ticker, filing_type, years=5):
            if filing_type == FilingType.K10:
                return [sample_filing(ticker=ticker, ftype="10-K")]
            else:
                return [sample_filing(ticker=ticker, ftype="10-Q")]

        with patch.object(collector, 'search_filings', side_effect=mock_search):
            results = await collector.batch_collect_10k_10q(["AAPL", "MSFT"], years=3)
            assert "AAPL" in results
            assert "MSFT" in results
            assert len(results["AAPL"]) == 2  # 1 10-K + 1 10-Q

    @pytest.mark.slow
    @pytest.mark.asyncio
    async def test_batch_collect_with_error(self, collector):
        async def mock_search(ticker, filing_type, years=5):
            if ticker == "ERR":
                raise Exception("Network error")
            return [sample_filing(ticker=ticker, ftype="10-K")]

        with patch.object(collector, 'search_filings', side_effect=mock_search):
            results = await collector.batch_collect_10k_10q(["AAPL", "ERR"])
            assert len(results["AAPL"]) >= 1
            assert results["ERR"] == []


# ===== collect_fundamentals 测试 =====

class TestCollectFundamentals:
    """一站式基本面拉取"""

    @pytest.mark.slow
    @pytest.mark.asyncio
    async def test_collect_fundamentals_mocked(self, collector):
        """通过 Mock edgartools / HTTP 拉取基本面"""
        mock_cik_resp = MagicMock()
        mock_cik_resp.status = 200
        mock_cik_resp.json = AsyncMock(return_value=MOCK_CIK_MAP)
        mock_cik_resp.__aenter__ = AsyncMock(return_value=mock_cik_resp)
        mock_cik_resp.__aexit__ = AsyncMock(return_value=None)

        mock_sub_resp = MagicMock()
        mock_sub_resp.status = 200
        mock_sub_resp.json = AsyncMock(return_value=MOCK_SUBMISSIONS_AAPL)
        mock_sub_resp.__aenter__ = AsyncMock(return_value=mock_sub_resp)
        mock_sub_resp.__aexit__ = AsyncMock(return_value=None)

        mock_facts_resp = MagicMock()
        mock_facts_resp.status = 200
        mock_facts_resp.json = AsyncMock(return_value=MOCK_COMPANY_FACTS_AAPL)
        mock_facts_resp.__aenter__ = AsyncMock(return_value=mock_facts_resp)
        mock_facts_resp.__aexit__ = AsyncMock(return_value=None)

        mock_session = MagicMock()
        mock_session.get = MagicMock(side_effect=[
            mock_cik_resp, mock_sub_resp,  # search_filings
            mock_facts_resp,               # parse_xbrl
        ])
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)

        with patch.object(collector, '_get_session', return_value=mock_session):
            collector._edgar_tools_available = False
            batch = await collector.collect_fundamentals(["AAPL"])
            assert batch.total_count == 1
