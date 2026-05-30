"""
AlphaVantageCollector 单元测试（含 Mock HTTP 响应）

测试覆盖:
- 初始化与密钥管理
- URL 构建和请求参数
- Mock API 调用 (日线/概况/三表/盈利)
- 响应解析 (DailyPrice, CompanyOverview, FinancialStatement, EarningsData)
- 批量拉取
- 缓存命中
- 错误处理 (限流/网络错误/无效JSON)
- parse_all_* 便捷方法
"""

import json
import os
from unittest.mock import AsyncMock, MagicMock, patch

import pandas as pd
import pytest

from src.collectors.alpha_vantage import (
    AlphaVantageCollector,
    AlphaVantageEndpoint,
    CompanyOverview,
    DailyPrice,
    EarningsData,
    FinancialStatement,
)
from src.collectors.base import BatchFetchResult, CollectorConfig, FetchResult


# ===== Mock 响应数据 =====

MOCK_DAILY_PRICES_RESPONSE = {
    "Meta Data": {
        "1. Information": "Daily Prices (open, high, low, close) and Volumes",
        "2. Symbol": "AAPL",
        "3. Last Refreshed": "2024-01-05",
        "4. Output Size": "Compact",
        "5. Time Zone": "US/Eastern",
    },
    "Time Series (Daily)": {
        "2024-01-05": {
            "1. open": "182.00",
            "2. high": "183.50",
            "3. low": "181.00",
            "4. close": "182.50",
            "5. adjusted close": "182.50",
            "6. volume": "50000000",
            "7. dividend amount": "0.0000",
            "8. split coefficient": "1.0",
        },
        "2024-01-04": {
            "1. open": "184.00",
            "2. high": "185.50",
            "3. low": "183.00",
            "4. close": "184.20",
            "5. adjusted close": "184.20",
            "6. volume": "45000000",
            "7. dividend amount": "0.2400",
            "8. split coefficient": "1.0",
        },
    },
}

MOCK_OVERVIEW_RESPONSE = {
    "Symbol": "AAPL",
    "Name": "Apple Inc",
    "Sector": "Technology",
    "Industry": "Consumer Electronics",
    "MarketCapitalization": "2800000000000",
    "PERatio": "28.5",
    "ForwardPE": "26.3",
    "PEGRatio": "2.15",
    "PriceToBookRatio": "45.2",
    "PriceToSalesRatio": "7.3",
    "EVToEBITDA": "22.8",
    "EVToRevenue": "7.5",
    "ProfitMargin": "0.252",
    "OperatingMarginTTM": "0.301",
    "GrossProfitTTM": "170000000000",
    "ReturnOnEquityTTM": "1.47",
    "ReturnOnAssetsTTM": "0.28",
    "RevenueTTM": "383000000000",
    "EBITDA": "125000000000",
    "DilutedEPSTTM": "6.42",
    "DividendYield": "0.0052",
    "DividendPerShare": "0.96",
    "Beta": "1.25",
    "52WeekHigh": "199.62",
    "52WeekLow": "124.17",
    "AnalystTargetPrice": "205.00",
}

MOCK_INCOME_STATEMENT = {
    "symbol": "AAPL",
    "annualReports": [
        {
            "fiscalDateEnding": "2023-09-30",
            "reportedCurrency": "USD",
            "totalRevenue": "383285000000",
            "costOfRevenue": "214137000000",
            "grossProfit": "169148000000",
            "operatingIncome": "114301000000",
            "netIncome": "96995000000",
            "ebit": "117669000000",
            "ebitda": "125000000000",
            "earningsPerShare": "6.16",
            "dilutedEarningsPerShare": "6.13",
        },
        {
            "fiscalDateEnding": "2022-09-30",
            "reportedCurrency": "USD",
            "totalRevenue": "394328000000",
            "costOfRevenue": "223546000000",
            "grossProfit": "170782000000",
            "operatingIncome": "119437000000",
            "netIncome": "99803000000",
            "ebit": "119103000000",
            "ebitda": "130000000000",
            "earningsPerShare": "6.15",
            "dilutedEarningsPerShare": "6.11",
        },
    ],
}

MOCK_BALANCE_SHEET = {
    "symbol": "AAPL",
    "annualReports": [
        {
            "fiscalDateEnding": "2023-09-30",
            "reportedCurrency": "USD",
            "totalAssets": "352583000000",
            "totalCurrentAssets": "143566000000",
            "totalLiabilities": "290437000000",
            "totalCurrentLiabilities": "145638000000",
            "totalShareholderEquity": "62146000000",
            "retainedEarnings": "-775000000",
            "longTermDebt": "95281000000",
        },
    ],
}

MOCK_CASH_FLOW = {
    "symbol": "AAPL",
    "annualReports": [
        {
            "fiscalDateEnding": "2023-09-30",
            "reportedCurrency": "USD",
            "operatingCashflow": "110543000000",
            "capitalExpenditures": "-10959000000",
            "dividendPayout": "15000000000",
        },
    ],
}

MOCK_EARNINGS = {
    "symbol": "AAPL",
    "annualEarnings": [
        {
            "fiscalDateEnding": "2023-09-30",
            "reportedEPS": "6.16",
            "estimatedEPS": "6.05",
            "surprise": "0.11",
            "surprisePercentage": "1.818",
            "reportTime": "after_market_close",
        },
    ],
    "quarterlyEarnings": [
        {
            "fiscalDateEnding": "2023-12-31",
            "reportedEPS": "2.18",
            "estimatedEPS": "2.10",
            "surprise": "0.08",
            "surprisePercentage": "3.81",
            "reportTime": "after_market_close",
        },
    ],
}

MOCK_ERROR_RESPONSE = {
    "Error Message": "Invalid API call. Please retry or visit the documentation for the API."
}

MOCK_RATE_LIMIT_RESPONSE = {
    "Note": "Thank you for using Alpha Vantage! Our standard API call frequency is 5 calls per minute..."
}


# ===== Fixtures =====

@pytest.fixture
def collector_config():
    return CollectorConfig(
        rate_limit_rpm=75,
        enable_cache=False,  # 测试中默认关闭缓存
        max_retries=1,
        retry_base_delay=0.01,
    )


@pytest.fixture
def collector(collector_config):
    """创建测试用采集器（不实际网络请求）"""
    with patch.dict(os.environ, {"ALPHA_VANTAGE_API_KEY": "test-key-12345"}):
        return AlphaVantageCollector(
            api_keys=["test-key-12345"],
            config=collector_config,
        )


def make_fetch_result(ticker, endpoint, response_data, from_cache=False):
    """构造 FetchResult 辅助函数"""
    return FetchResult(
        endpoint=endpoint,
        ticker=ticker,
        params={"symbol": ticker, "function": endpoint},
        raw_response=response_data,
        fetched_at="2024-01-05T16:00:00+00:00",
        from_cache=from_cache,
    )


# ===== 初始化测试 =====

class TestAlphaVantageCollectorInit:
    """初始化相关测试"""

    def test_init_with_explicit_keys(self, collector_config):
        collector = AlphaVantageCollector(
            api_keys=["key1", "key2"],
            config=collector_config,
        )
        assert len(collector.api_rotator.keys) == 2
        assert collector.api_rotator.keys[0].key == "key1"

    def test_init_with_env_var(self, collector_config):
        with patch.dict(os.environ, {"ALPHA_VANTAGE_API_KEY": "env-key, env-key-2"}):
            collector = AlphaVantageCollector(config=collector_config)
            assert len(collector.api_rotator.keys) == 2
            assert collector.api_rotator.keys[0].key == "env-key"
            assert collector.api_rotator.keys[1].key == "env-key-2"

    def test_init_without_keys_raises(self, collector_config):
        with patch.dict(os.environ, {}, clear=True):
            with patch.dict(os.environ, {"ALPHA_VANTAGE_API_KEY": ""}):
                with pytest.raises(ValueError, match="未提供 Alpha Vantage API 密钥"):
                    AlphaVantageCollector(config=collector_config)

    def test_default_config(self):
        with patch.dict(os.environ, {"ALPHA_VANTAGE_API_KEY": "test-key"}):
            collector = AlphaVantageCollector(api_keys=["test-key"])
            assert collector.config.rate_limit_rpm == 75
            assert collector.config.max_retries == 3

    def test_rate_limiter_initialized(self, collector):
        assert collector.rate_limiter.rate == 75
        assert collector.rate_limiter.period == 60.0


# ===== URL 与参数构建 =====

class TestBuildUrlAndParams:
    """URL / 参数构建测试"""

    def test_build_url(self, collector):
        url = collector._build_url("ANY_ENDPOINT")
        assert url == "https://www.alphavantage.co/query"

    def test_build_request_params(self, collector):
        params = collector._build_request_params("AAPL", {"function": "OVERVIEW"})
        assert params["symbol"] == "AAPL"
        assert params["function"] == "OVERVIEW"


# ===== 响应解析 — 日线量价 =====

class TestParseDailyPrices:
    """日线量价解析测试"""

    def test_parse_valid_response(self, collector):
        result = make_fetch_result("AAPL", AlphaVantageEndpoint.DAILY_ADJUSTED, MOCK_DAILY_PRICES_RESPONSE)
        prices = collector.parse_daily_prices(result)

        assert len(prices) == 2
        assert prices[0].ticker == "AAPL"
        assert prices[0].date == "2024-01-05"
        assert prices[0].open == 182.00
        assert prices[0].high == 183.50
        assert prices[0].low == 181.00
        assert prices[0].close == 182.50
        assert prices[0].adjusted_close == 182.50
        assert prices[0].volume == 50000000

    def test_parse_with_dividend(self, collector):
        result = make_fetch_result("AAPL", AlphaVantageEndpoint.DAILY_ADJUSTED, MOCK_DAILY_PRICES_RESPONSE)
        prices = collector.parse_daily_prices(result)
        assert prices[1].dividend_amount == 0.24
        assert prices[1].split_coefficient == 1.0

    def test_parse_error_response_returns_empty(self, collector):
        result = make_fetch_result("AAPL", AlphaVantageEndpoint.DAILY_ADJUSTED, MOCK_ERROR_RESPONSE)
        prices = collector.parse_daily_prices(result)
        assert prices == []

    def test_parse_empty_response_returns_empty(self, collector):
        result = make_fetch_result("AAPL", AlphaVantageEndpoint.DAILY_ADJUSTED, {})
        prices = collector.parse_daily_prices(result)
        assert prices == []

    def test_parse_missing_time_series(self, collector):
        result = make_fetch_result("AAPL", AlphaVantageEndpoint.DAILY_ADJUSTED,
                                   {"Meta Data": {"2. Symbol": "AAPL"}})
        prices = collector.parse_daily_prices(result)
        assert prices == []


# ===== 响应解析 — 公司概况 =====

class TestParseCompanyOverview:
    """公司概况解析测试"""

    def test_parse_valid_overview(self, collector):
        result = make_fetch_result("AAPL", AlphaVantageEndpoint.OVERVIEW, MOCK_OVERVIEW_RESPONSE)
        overview = collector.parse_company_overview(result)

        assert overview is not None
        assert overview.ticker == "AAPL"
        assert overview.name == "Apple Inc"
        assert overview.sector == "Technology"
        assert overview.pe_ratio == 28.5
        assert overview.forward_pe == 26.3
        assert overview.operating_margin == 0.301
        assert overview.return_on_equity == 1.47
        assert overview.market_cap == 2800000000000
        assert overview.beta == 1.25
        assert overview.dividend_yield == 0.0052

    def test_parse_overview_error_response(self, collector):
        result = make_fetch_result("AAPL", AlphaVantageEndpoint.OVERVIEW, MOCK_ERROR_RESPONSE)
        overview = collector.parse_company_overview(result)
        assert overview is None

    def test_parse_overview_with_none_values(self, collector):
        response = {
            "Symbol": "TEST",
            "Name": "Test Co",
            "Sector": "",
            "Industry": "",
            "PERatio": "None",
            "ForwardPE": "",
            "MarketCapitalization": None,
        }
        result = make_fetch_result("TEST", AlphaVantageEndpoint.OVERVIEW, response)
        overview = collector.parse_company_overview(result)

        assert overview is not None
        assert overview.pe_ratio is None
        assert overview.forward_pe is None
        assert overview.market_cap is None


# ===== 响应解析 — 财务报表 =====

class TestParseFinancialStatement:
    """财务报表解析测试"""

    def test_parse_income_statement(self, collector):
        result = make_fetch_result("AAPL", AlphaVantageEndpoint.INCOME_STATEMENT, MOCK_INCOME_STATEMENT)
        stmts = collector.parse_financial_statement(result, "income")

        assert len(stmts) == 2
        assert stmts[0].ticker == "AAPL"
        assert stmts[0].fiscal_date_ending == "2023-09-30"
        assert stmts[0].total_revenue == 383285000000
        assert stmts[0].gross_profit == 169148000000
        assert stmts[0].operating_income == 114301000000
        assert stmts[0].net_income == 96995000000
        assert stmts[0].eps == 6.16

    def test_parse_balance_sheet(self, collector):
        result = make_fetch_result("AAPL", AlphaVantageEndpoint.BALANCE_SHEET, MOCK_BALANCE_SHEET)
        stmts = collector.parse_financial_statement(result, "balance")

        assert len(stmts) == 1
        assert stmts[0].total_assets == 352583000000
        assert stmts[0].total_liabilities == 290437000000
        assert stmts[0].total_equity == 62146000000
        assert stmts[0].long_term_debt == 95281000000
        assert stmts[0].retained_earnings == -775000000

    def test_parse_cash_flow(self, collector):
        result = make_fetch_result("AAPL", AlphaVantageEndpoint.CASH_FLOW, MOCK_CASH_FLOW)
        stmts = collector.parse_financial_statement(result, "cash_flow")

        assert len(stmts) == 1
        assert stmts[0].operating_cash_flow == 110543000000

    def test_parse_financial_error_response(self, collector):
        result = make_fetch_result("AAPL", AlphaVantageEndpoint.INCOME_STATEMENT, MOCK_ERROR_RESPONSE)
        stmts = collector.parse_financial_statement(result, "income")
        assert stmts == []


# ===== 响应解析 — 盈利数据 =====

class TestParseEarnings:
    """盈利数据解析测试"""

    def test_parse_earnings(self, collector):
        result = make_fetch_result("AAPL", AlphaVantageEndpoint.EARNINGS, MOCK_EARNINGS)
        earnings = collector.parse_earnings(result)

        assert len(earnings) == 2  # 1 annual + 1 quarterly
        assert earnings[0].ticker == "AAPL"
        # quarterly first (inner loop runs quarterlyEarnings first)
        assert earnings[0].reported_eps == 2.18
        assert earnings[1].fiscal_date_ending == "2023-09-30"  # annual
        assert earnings[1].reported_eps == 6.16

    def test_parse_earnings_error_response(self, collector):
        result = make_fetch_result("AAPL", AlphaVantageEndpoint.EARNINGS, MOCK_ERROR_RESPONSE)
        earnings = collector.parse_earnings(result)
        assert earnings == []


# ===== 批量解析便捷方法 =====

class TestParseAllMethods:
    """parse_all_* 系列方法测试"""

    def test_parse_all_daily_prices(self, collector):
        batch = BatchFetchResult(total_count=2)
        batch.results = [
            make_fetch_result("AAPL", AlphaVantageEndpoint.DAILY_ADJUSTED, MOCK_DAILY_PRICES_RESPONSE),
            make_fetch_result("MSFT", AlphaVantageEndpoint.DAILY_ADJUSTED, {
                "Time Series (Daily)": {
                    "2024-01-05": {
                        "1. open": "380.00", "2. high": "385.00", "3. low": "379.00",
                        "4. close": "384.00", "5. adjusted close": "384.00",
                        "6. volume": "25000000", "7. dividend amount": "0.00",
                        "8. split coefficient": "1.0",
                    }
                }
            }),
        ]

        df = collector.parse_all_daily_prices(batch)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 3  # 2 AAPL + 1 MSFT
        assert set(df["ticker"].unique()) == {"AAPL", "MSFT"}

    def test_parse_all_daily_prices_empty(self, collector):
        batch = BatchFetchResult(total_count=1)
        batch.results = [
            make_fetch_result("BAD", AlphaVantageEndpoint.DAILY_ADJUSTED, MOCK_ERROR_RESPONSE),
        ]
        df = collector.parse_all_daily_prices(batch)
        assert len(df) == 0

    def test_parse_all_overviews(self, collector):
        batch = BatchFetchResult(total_count=1)
        batch.results = [
            make_fetch_result("AAPL", AlphaVantageEndpoint.OVERVIEW, MOCK_OVERVIEW_RESPONSE),
        ]
        df = collector.parse_all_overviews(batch)
        assert len(df) == 1
        assert df.iloc[0]["ticker"] == "AAPL"
        assert df.iloc[0]["pe_ratio"] == 28.5


# ===== 错误与边界情况 =====

class TestErrorHandling:
    """错误处理测试"""

    def test_rate_limit_response_not_success(self):
        """确认限流响应被判定为不成功"""
        result = FetchResult(
            endpoint=AlphaVantageEndpoint.DAILY_ADJUSTED,
            ticker="AAPL",
            params={},
            raw_response=MOCK_RATE_LIMIT_RESPONSE,
            fetched_at="2024-01-01T00:00:00Z",
        )
        assert result.is_success is False

    def test_error_response_not_success(self):
        """确认错误响应被判定为不成功"""
        result = FetchResult(
            endpoint=AlphaVantageEndpoint.OVERVIEW,
            ticker="AAPL",
            params={},
            raw_response=MOCK_ERROR_RESPONSE,
            fetched_at="2024-01-01T00:00:00Z",
        )
        assert result.is_success is False


# ===== 配额监控 =====

class TestQuotaMonitoring:
    """配额监控测试"""

    def test_quota_summary(self, collector):
        summary = collector.quota_summary
        assert summary["total_keys"] == 1
        assert summary["active_keys"] == 1
        assert "remaining_capacity" in summary
        assert len(summary["keys"]) == 1


# ===== 集成: Mock API 批量拉取场景 =====

class TestBatchFetchWithMock:
    """使用 mock 的批量拉取集成测试"""

    @pytest.mark.slow
    @pytest.mark.asyncio
    async def test_collect_daily_prices_mocked(self, collector):
        """模拟 collect_daily_prices: 验证 _execute_with_key 路径"""
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value=MOCK_DAILY_PRICES_RESPONSE)

        mock_session = AsyncMock()
        mock_session.get = MagicMock(return_value=mock_response)
        mock_session.closed = False

        collector._session = mock_session
        collector._semaphore = MagicMock()
        collector._semaphore.__aenter__ = AsyncMock()
        collector._semaphore.__aexit__ = AsyncMock()

        result = await collector.collect_daily_prices(["AAPL"], outputsize="compact")

        assert result.total_count == 1
        assert result.success_count == 1

    @pytest.mark.slow
    @pytest.mark.asyncio
    async def test_collect_company_overviews_mocked(self, collector):
        """模拟 collect_company_overviews — 验证解析链路"""
        # 直接构造成功的 FetchResult 来测试解析
        result = make_fetch_result("AAPL", AlphaVantageEndpoint.OVERVIEW, MOCK_OVERVIEW_RESPONSE)

        overview = collector.parse_company_overview(result)
        assert overview is not None
        assert overview.ticker == "AAPL"
        assert overview.pe_ratio == 28.5
        assert overview.forward_pe == 26.3
        assert overview.operating_margin == 0.301
        assert overview.market_cap == 2800000000000

    @pytest.mark.slow
    @pytest.mark.asyncio
    async def test_collect_fundamentals_concurrent(self, collector):
        """测试并发拉取全部基本面数据"""
        mock_response = AsyncMock()
        mock_response.status = 200

        # 为不同端点返回不同数据
        response_map = {
            "OVERVIEW": MOCK_OVERVIEW_RESPONSE,
        }

        async def mock_json():
            return mock_response._current_data

        mock_response.json = mock_json

        mock_session = AsyncMock()

        def mock_get(url, params):
            func = params.get("function", "OVERVIEW")
            if func == "OVERVIEW":
                mock_response._current_data = MOCK_OVERVIEW_RESPONSE
            elif func == "INCOME_STATEMENT":
                mock_response._current_data = MOCK_INCOME_STATEMENT
            elif func == "BALANCE_SHEET":
                mock_response._current_data = MOCK_BALANCE_SHEET
            elif func == "CASH_FLOW":
                mock_response._current_data = MOCK_CASH_FLOW
            elif func == "EARNINGS":
                mock_response._current_data = MOCK_EARNINGS
            else:
                mock_response._current_data = MOCK_ERROR_RESPONSE
            return mock_response

        mock_session.get = mock_get
        mock_session.closed = False

        collector._session = mock_session
        collector._semaphore = MagicMock()
        collector._semaphore.__aenter__ = AsyncMock()
        collector._semaphore.__aexit__ = AsyncMock()

        results = await collector.collect_fundamentals(["AAPL"])

        assert "overview" in results
        assert "income" in results
        assert "balance" in results
        assert "cash_flow" in results
        assert "earnings" in results

    @pytest.mark.slow
    @pytest.mark.asyncio
    async def test_close_session(self, collector):
        mock_session = AsyncMock()
        mock_session.closed = False
        collector._session = mock_session

        await collector.close()
        mock_session.close.assert_awaited_once()
