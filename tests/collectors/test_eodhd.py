"""
EODHDCollector 单元测试（含 Mock HTTP 响应）

测试覆盖:
- 初始化与密钥管理
- 基本面解析 (EODHDFundamentals)
- 企业行动解析 (拆分/股息 → CorporateAction)
- 宏观经济指标解析 (MacroDataPoint)
- 交叉验证 (CrossValidationResult)
- 批量解析 (parse_all_* → DataFrame)
- Mock HTTP 批量和并发拉取
"""

import json
import os
from unittest.mock import AsyncMock, MagicMock, patch

import pandas as pd
import pytest

from src.collectors.alpha_vantage import CompanyOverview, FinancialStatement
from src.collectors.base import BatchFetchResult, CollectorConfig, FetchResult
from src.collectors.eodhd import (
    CorporateAction,
    Country,
    CrossValidationResult,
    EODHDCollector,
    EODHDEndpoint,
    EODHDFundamentals,
    MacroDataPoint,
    MacroIndicator,
)


# ===== Mock 响应数据 =====

MOCK_FUNDAMENTALS_AAPL = {
    "Code": "AAPL",
    "General": {
        "Code": "AAPL",
        "Name": "Apple Inc.",
        "Exchange": "NASDAQ",
        "CurrencyCode": "USD",
        "Sector": "Technology",
        "Industry": "Consumer Electronics",
        "CountryName": "USA",
        "ISIN": "US0378331005",
    },
    "Highlights": {
        "MarketCapitalization": 2800000000000,
        "PERatio": 28.5,
        "PEGRatio": 2.15,
        "RevenueTTM": 383285000000,
        "GrossProfitTTM": 169148000000,
        "EBITDA": 125000000000,
        "NetIncomeTTM": 96995000000,
        "FreeCashFlowTTM": 99500000000,
        "DilutedEpsTTM": 6.42,
        "ProfitMargin": 0.253,
        "OperatingMargin": 0.301,
        "GrossMargin": 0.441,
        "ReturnOnEquity": 1.47,
        "ReturnOnAssets": 0.28,
        "ReturnOnInvestedCapital": 0.56,
        "TotalAssets": 352583000000,
        "TotalDebt": 111088000000,
        "TotalEquity": 62146000000,
        "CurrentRatio": 0.99,
        "DebtToEquity": 1.79,
        "DividendYield": 0.0052,
        "DividendShare": 0.96,
        "PayoutRatio": 0.156,
        "BuybackYield": 0.032,
        "RevenueGrowth": -0.028,
        "EarningsGrowth": -0.029,
        "AnalystTargetPrice": 205.0,
    },
    "Valuation": {
        "TrailingPE": 28.5,
        "ForwardPE": 26.3,
        "PriceSalesTTM": 7.32,
        "PriceBookMRQ": 45.2,
        "EVToEBITDA": 22.8,
        "EVToRevenue": 7.5,
        "EnterpriseValue": 2850000000000,
    },
    "Technicals": {
        "Beta": 1.25,
        "52WeekHigh": 199.62,
        "52WeekLow": 124.17,
    },
    "SharesStats": {
        "SharesOutstanding": 15500000000,
    },
}

MOCK_SPLITS_AAPL = [
    {"date": "2020-08-31", "split": "4-for-1"},
    {"date": "2014-06-09", "split": "7-for-1"},
    {"date": "2005-02-28", "split": "2-for-1"},
]

MOCK_DIVIDENDS_AAPL = [
    {"date": "2024-02-09", "value": 0.24, "type": "Cash", "currency": "USD",
     "declarationDate": "2024-02-01", "recordDate": "2024-02-12", "paymentDate": "2024-02-15"},
    {"date": "2023-11-10", "value": 0.24, "type": "Cash", "currency": "USD",
     "declarationDate": "2023-11-02", "recordDate": "2023-11-13", "paymentDate": "2023-11-16"},
    {"date": "2023-08-11", "value": 0.24, "type": "Cash", "currency": "USD"},
]

MOCK_MACRO_GDP = [
    {"Date": "2024-01-01", "Value": 28200, "Unit": "Billion USD", "Frequency": "quarterly"},
    {"Date": "2023-10-01", "Value": 27700, "Unit": "Billion USD", "Frequency": "quarterly"},
    {"Date": "2023-07-01", "Value": 27200, "Unit": "Billion USD", "Frequency": "quarterly"},
]

MOCK_CPI = [
    {"date": "2024-01-01", "value": 3.1, "unit": "%", "frequency": "monthly"},
    {"date": "2023-12-01", "value": 3.4, "unit": "%", "frequency": "monthly"},
]

MOCK_ERROR_MSG = "Could not find the requested file."


# ===== Fixtures =====

@pytest.fixture
def collector():
    """创建测试用 EODHD 采集器"""
    with patch.dict(os.environ, {"EODHD_API_KEY": "test-eodhd-token"}):
        config = CollectorConfig(rate_limit_rpm=100, enable_cache=False, max_retries=1)
        return EODHDCollector(api_keys=["test-eodhd-token"], config=config)


def make_fetch_result(ticker, endpoint, response_data, from_cache=False):
    return FetchResult(
        endpoint=endpoint,
        ticker=ticker,
        params={},
        raw_response=response_data,
        fetched_at="2024-01-05T16:00:00+00:00",
        from_cache=from_cache,
    )


# ===== 初始化测试 =====

class TestEODHDCollectorInit:
    """初始化相关测试"""

    def test_init_with_explicit_key(self):
        config = CollectorConfig(rate_limit_rpm=100, enable_cache=False)
        collector = EODHDCollector(api_keys=["my-token"], config=config)
        assert len(collector.api_rotator.keys) == 1
        assert collector.api_rotator.keys[0].key == "my-token"

    def test_init_with_env_var(self):
        with patch.dict(os.environ, {"EODHD_API_KEY": "env-token, env-token-2"}):
            config = CollectorConfig(rate_limit_rpm=100, enable_cache=False)
            collector = EODHDCollector(config=config)
            assert len(collector.api_rotator.keys) == 2

    def test_init_without_keys_raises(self):
        with patch.dict(os.environ, {"EODHD_API_KEY": ""}):
            with pytest.raises(ValueError, match="未提供 EODHD API 密钥"):
                EODHDCollector(config=CollectorConfig(enable_cache=False))

    def test_build_url(self, collector):
        url = collector._build_url("fundamentals/AAPL")
        assert url == "https://eodhistoricaldata.com/api/fundamentals/AAPL"

    def test_build_request_params(self, collector):
        params = collector._build_request_params("AAPL", {"filter": "extended"})
        assert params == {"filter": "extended"}


# ===== 基本面解析 =====

class TestParseFundamentals:
    """EODHDFundamentals 解析测试"""

    def test_parse_full_fundamentals(self, collector):
        result = make_fetch_result("AAPL", "fundamentals", MOCK_FUNDAMENTALS_AAPL)
        f = collector.parse_fundamentals(result)

        assert f is not None
        assert f.ticker == "AAPL"
        assert f.name == "Apple Inc."
        assert f.exchange == "NASDAQ"
        assert f.sector == "Technology"
        assert f.industry == "Consumer Electronics"
        assert f.country == "USA"

        # 估值
        assert f.market_cap == 2800000000000
        assert f.pe_ratio == 28.5
        assert f.forward_pe == 26.3
        assert f.peg_ratio == 2.15
        assert f.ev_to_ebitda == 22.8

        # 盈利能力
        assert f.profit_margin == 0.253
        assert f.operating_margin == 0.301
        assert f.return_on_equity == 1.47

        # 财务
        assert f.revenue_ttm == 383285000000
        assert f.net_income_ttm == 96995000000
        assert f.diluted_eps_ttm == 6.42

        # 资产负债
        assert f.total_assets == 352583000000
        assert f.debt_to_equity == 1.79

        # 股息
        assert f.dividend_yield == 0.0052
        assert f.dividend_per_share == 0.96

        # 技术指标
        assert f.beta == 1.25
        assert f._52_week_high == 199.62
        assert f._52_week_low == 124.17

        # 原始数据
        assert len(f.raw_data) > 0

    def test_parse_empty_response(self, collector):
        result = make_fetch_result("ZZZ", "fundamentals", {})
        f = collector.parse_fundamentals(result)
        assert f is None

    def test_parse_error_response(self, collector):
        result = FetchResult(
            endpoint="fundamentals", ticker="ZZZ", params={},
            raw_response="error: not found",
            fetched_at="2024-01-01T00:00:00Z",
        )
        f = collector.parse_fundamentals(result)
        assert f is None

    def test_parse_minimal_fundamentals(self, collector):
        """只有部分字段的基本面数据"""
        minimal = {"Code": "TEST", "General": {"Name": "Test Co"}, "Highlights": {}}
        result = make_fetch_result("TEST", "fundamentals", minimal)
        f = collector.parse_fundamentals(result)
        assert f is not None
        assert f.ticker == "TEST"
        assert f.pe_ratio is None
        assert f.market_cap is None


# ===== 企业行动 — 拆分 =====

class TestParseSplits:
    """拆分解析测试"""

    def test_parse_splits_list(self, collector):
        result = make_fetch_result("AAPL", "splits", MOCK_SPLITS_AAPL)
        actions = collector.parse_splits(result)

        assert len(actions) == 3
        assert all(a.action_type == "split" for a in actions)

        # "4-for-1" 拆分
        assert actions[0].date == "2020-08-31"
        assert actions[0].split_from == 4.0
        assert actions[0].split_to == 1.0
        assert actions[0].value == 4.0

        # "7-for-1"
        assert actions[1].value == 7.0

        # "2-for-1"
        assert actions[2].split_from == 2.0
        assert actions[2].split_to == 1.0

    def test_parse_splits_empty(self, collector):
        result = make_fetch_result("NO_SPLIT", "splits", [])
        actions = collector.parse_splits(result)
        assert actions == []

    def test_parse_splits_error(self, collector):
        result = make_fetch_result("ERR", "splits", {"error": "not found"})
        actions = collector.parse_splits(result)
        assert actions == []


# ===== 企业行动 — 股息 =====

class TestParseDividends:
    """股息解析测试"""

    def test_parse_dividends_list(self, collector):
        result = make_fetch_result("AAPL", "dividends", MOCK_DIVIDENDS_AAPL)
        actions = collector.parse_dividends(result)

        assert len(actions) == 3
        assert all(a.action_type == "dividend" for a in actions)
        assert actions[0].value == 0.24
        assert actions[0].dividend_type == "Cash"
        assert actions[0].currency == "USD"
        assert actions[0].declaration_date == "2024-02-01"
        assert actions[0].record_date == "2024-02-12"
        assert actions[0].payment_date == "2024-02-15"

    def test_parse_dividends_dict_format(self, collector):
        """字典格式: {date: value}"""
        response = {"2024-01-15": 0.30, "2023-10-15": 0.28}
        result = make_fetch_result("DIV", "dividends", response)
        actions = collector.parse_dividends(result)

        assert len(actions) == 2
        assert actions[0].value == 0.30
        assert actions[1].value == 0.28

    def test_parse_dividends_empty(self, collector):
        result = make_fetch_result("NONE", "dividends", [])
        actions = collector.parse_dividends(result)
        assert actions == []


# ===== 宏观经济指标 =====

class TestParseMacro:
    """宏观经济指标解析测试"""

    def test_parse_macro_gdp(self, collector):
        result = make_fetch_result(MacroIndicator.USA_GDP.value, "macro", MOCK_MACRO_GDP)
        points = collector.parse_macro_indicator(result)

        assert len(points) == 3
        assert points[0].indicator == MacroIndicator.USA_GDP.value
        assert points[0].value == 28200
        assert points[0].unit == "Billion USD"
        assert points[0].frequency == "quarterly"

    def test_parse_macro_cpi(self, collector):
        result = make_fetch_result(MacroIndicator.USA_CPI.value, "macro", MOCK_CPI)
        points = collector.parse_macro_indicator(result)

        assert len(points) == 2
        assert points[0].value == 3.1
        assert points[0].unit == "%"

    def test_parse_macro_empty(self, collector):
        result = make_fetch_result("NONE", "macro", [])
        points = collector.parse_macro_indicator(result)
        assert points == []


# ===== 交叉验证 =====

class TestCrossValidation:
    """交叉验证逻辑测试"""

    def test_cross_validate_consistent(self, collector):
        """两源数据一致 → ok"""
        av = CompanyOverview(
            ticker="AAPL", name="Apple", sector="Tech", industry="Electronics",
            pe_ratio=28.5, forward_pe=26.3, market_cap=2800000000000,
        )
        eodhd = EODHDFundamentals(
            ticker="AAPL", pe_ratio=28.5, forward_pe=26.3, market_cap=2800000000000,
        )

        results = collector.cross_validate_fundamentals("AAPL", av_overview=av, eodhd_fundamentals=eodhd)

        for r in results:
            assert r.status == "ok"
            assert r.deviation_pct < 5.0

    def test_cross_validate_slight_deviation(self, collector):
        """轻微偏差 → warning"""
        av = CompanyOverview(
            ticker="MSFT", name="MSFT", sector="Tech", industry="Software",
            pe_ratio=35.0,
        )
        eodhd = EODHDFundamentals(ticker="MSFT", pe_ratio=37.8)  # ~8% higher

        results = collector.cross_validate_fundamentals("MSFT", av_overview=av, eodhd_fundamentals=eodhd)
        pe_result = next(r for r in results if r.field == "pe_ratio")
        assert pe_result.status == "warning"
        assert 5 <= pe_result.deviation_pct < 10

    def test_cross_validate_conflict(self, collector):
        """显著偏差 → conflict (以 EODHD 为准)"""
        av = CompanyOverview(
            ticker="XYZ", name="XYZ", sector="Tech", industry="Tech",
            pe_ratio=100.0,
        )
        eodhd = EODHDFundamentals(ticker="XYZ", pe_ratio=70.0)  # ~30% lower

        results = collector.cross_validate_fundamentals("XYZ", av_overview=av, eodhd_fundamentals=eodhd)
        pe_result = next(r for r in results if r.field == "pe_ratio")
        assert pe_result.status == "conflict"
        assert pe_result.eodhd_priority is True
        assert pe_result.resolved_value == 70.0

    def test_cross_validate_none_values_skipped(self, collector):
        """None 值的字段应被跳过"""
        av = CompanyOverview(ticker="ABC", name="ABC", sector="Fin", industry="Bank")
        eodhd = EODHDFundamentals(ticker="ABC")
        results = collector.cross_validate_fundamentals("ABC", av_overview=av, eodhd_fundamentals=eodhd)
        assert len(results) == 0  # 所有值都是 None

    def test_cross_validate_with_financials(self, collector):
        """带财务报表的交叉验证"""
        av_fs = [
            FinancialStatement(
                ticker="AAPL", statement_type="income", fiscal_date_ending="2023-09-30",
                reported_currency="USD", total_revenue=383285000000,
                gross_profit=169148000000, net_income=96995000000,
            )
        ]
        eodhd = EODHDFundamentals(
            ticker="AAPL", revenue_ttm=383285000000,
            gross_profit_ttm=169148000000, net_income_ttm=96995000000,
        )

        results = collector.cross_validate_fundamentals(
            "AAPL", av_financials=av_fs, eodhd_fundamentals=eodhd,
        )

        assert len(results) == 3
        assert all(r.status == "ok" for r in results)


# ===== 批量解析 =====

class TestParseAllMethods:
    """parse_all_* 便捷方法测试"""

    def test_parse_all_fundamentals(self, collector):
        batch = BatchFetchResult(total_count=2)
        batch.results = [
            make_fetch_result("AAPL", "fundamentals", MOCK_FUNDAMENTALS_AAPL),
            make_fetch_result("MSFT", "fundamentals", {
                "Code": "MSFT",
                "General": {"Name": "Microsoft", "Sector": "Technology"},
                "Highlights": {"PERatio": 35.0, "MarketCapitalization": 3000000000000},
            }),
        ]

        df = collector.parse_all_fundamentals(batch)
        assert len(df) == 2
        assert set(df["ticker"]) == {"AAPL", "MSFT"}

    def test_parse_all_splits(self, collector):
        batch = BatchFetchResult(total_count=1)
        batch.results = [make_fetch_result("AAPL", "splits", MOCK_SPLITS_AAPL)]

        df = collector.parse_all_splits(batch)
        assert len(df) == 3
        assert df.iloc[0]["action_type"] == "split"

    def test_parse_all_dividends(self, collector):
        batch = BatchFetchResult(total_count=1)
        batch.results = [make_fetch_result("AAPL", "dividends", MOCK_DIVIDENDS_AAPL)]

        df = collector.parse_all_dividends(batch)
        assert len(df) == 3
        assert all(df["action_type"] == "dividend")

    def test_parse_all_macro(self, collector):
        batch = BatchFetchResult(total_count=2)
        batch.results = [
            make_fetch_result(MacroIndicator.USA_GDP.value, "macro", MOCK_MACRO_GDP),
            make_fetch_result(MacroIndicator.USA_CPI.value, "macro", MOCK_CPI),
        ]

        df = collector.parse_all_macro(batch)
        assert len(df) == 5  # 3 GDP + 2 CPI

    def test_parse_all_empty(self, collector):
        batch = BatchFetchResult(total_count=1)
        # EODHD 错误响应是纯文本，不是 JSON 对象
        batch.results = [make_fetch_result("ERR", "fundamentals", "Could not find the requested file.")]
        df = collector.parse_all_fundamentals(batch)
        assert len(df) == 0


# ===== 枚举与常量 =====

class TestEnums:
    """枚举与常量测试"""

    def test_macro_indicator_values(self):
        assert MacroIndicator.USA_GDP.value == "USA_GDP"
        assert MacroIndicator.USA_CPI.value == "USA_CPI"
        assert MacroIndicator.USA_UNEMPLOYMENT_RATE.value == "USA_UNEMPLOYMENT_RATE"
        assert MacroIndicator.USA_FED_FUNDS_RATE.value == "USA_FED_FUNDS_RATE"
        assert MacroIndicator.USA_NONFARM_PAYROLLS.value == "USA_NONFARM_PAYROLLS"

    def test_country_codes(self):
        assert Country.US == "USA"
        assert Country.UK == "GBR"
        assert Country.CN == "CHN"
        assert Country.JP == "JPN"

    def test_endpoint_constants(self):
        assert EODHDEndpoint.FUNDAMENTALS == "fundamentals"
        assert EODHDEndpoint.SPLITS == "splits"
        assert EODHDEndpoint.DIVIDENDS == "divs"
        assert EODHDEndpoint.MACRO_INDICATOR == "macro-indicator"


# ===== Mock HTTP 集成测试 =====

class TestMockedAPICalls:
    """使用 Mock 的 API 调用集成测试"""

    @pytest.mark.slow
    @pytest.mark.asyncio
    async def test_collect_fundamentals_mocked(self, collector):
        mock_resp = AsyncMock()
        mock_resp.status = 200
        mock_resp.json = AsyncMock(return_value=MOCK_FUNDAMENTALS_AAPL)

        mock_session = AsyncMock()
        mock_session.get = AsyncMock(return_value=mock_resp)
        mock_session.closed = False

        collector._session = mock_session
        collector._semaphore = MagicMock()
        collector._semaphore.__aenter__ = AsyncMock()
        collector._semaphore.__aexit__ = AsyncMock()

        result = await collector.collect_fundamentals(["AAPL"])
        assert result.total_count == 1
        assert result.success_count == 1

    @pytest.mark.slow
    @pytest.mark.asyncio
    async def test_collect_corporate_actions_concurrent(self, collector):
        """测试企业行动并发拉取"""
        mock_resp_splits = AsyncMock()
        mock_resp_splits.status = 200
        mock_resp_splits.json = AsyncMock(return_value=MOCK_SPLITS_AAPL)

        mock_resp_divs = AsyncMock()
        mock_resp_divs.status = 200
        mock_resp_divs.json = AsyncMock(return_value=MOCK_DIVIDENDS_AAPL)

        call_count = [0]

        async def mock_get(url, params):
            call_count[0] += 1
            mock_resp = AsyncMock()
            mock_resp.status = 200
            if "splits" in url:
                mock_resp.json = AsyncMock(return_value=MOCK_SPLITS_AAPL)
            else:
                mock_resp.json = AsyncMock(return_value=MOCK_DIVIDENDS_AAPL)
            return mock_resp

        mock_session = AsyncMock()
        mock_session.get = mock_get
        mock_session.closed = False

        collector._session = mock_session
        collector._semaphore = MagicMock()
        collector._semaphore.__aenter__ = AsyncMock()
        collector._semaphore.__aexit__ = AsyncMock()

        results = await collector.collect_corporate_actions(["AAPL"])
        assert "splits" in results
        assert "dividends" in results

    @pytest.mark.slow
    @pytest.mark.asyncio
    async def test_close_session(self, collector):
        mock_session = AsyncMock()
        mock_session.closed = False
        collector._session = mock_session
        await collector.close()
        mock_session.close.assert_awaited_once()


# ===== 配额监控 =====

class TestQuotaMonitoring:
    def test_quota_summary(self, collector):
        summary = collector.quota_summary
        assert summary["total_keys"] == 1
        assert "remaining_capacity" in summary
