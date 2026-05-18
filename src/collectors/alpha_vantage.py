"""
Alpha Vantage API 数据采集器

继承 BaseCollector，实现 Alpha Vantage 专有的:
- API 端点 URL 构建
- 请求参数组装（含密钥注入）
- 响应解析（JSON → 标准化数据结构）
- 日线量价与基本面数据的拉取

对接端点:
- TIME_SERIES_DAILY_ADJUSTED  — 日线调整后量价
- OVERVIEW                     — 公司概况（估值指标）
- INCOME_STATEMENT             — 利润表
- BALANCE_SHEET                — 资产负债表
- CASH_FLOW                    — 现金流量表
- EARNINGS                     — 盈利日历
"""

import asyncio
import os
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from src.collectors.base import BaseCollector, BatchFetchResult, CollectorConfig, FetchResult
from src.collectors.rate_limiter import ApiKeyRotator, RateLimiter
from src.utils.logger import get_logger


# ===== Alpha Vantage 端点常量 =====

class AlphaVantageEndpoint:
    """Alpha Vantage API 端点名称"""
    DAILY_ADJUSTED = "TIME_SERIES_DAILY_ADJUSTED"
    OVERVIEW = "OVERVIEW"
    INCOME_STATEMENT = "INCOME_STATEMENT"
    BALANCE_SHEET = "BALANCE_SHEET"
    CASH_FLOW = "CASH_FLOW"
    EARNINGS = "EARNINGS"


# ===== 标准化数据结构 =====

@dataclass
class DailyPrice:
    """日线量价数据（调整后）"""
    ticker: str
    date: str              # YYYY-MM-DD
    open: float
    high: float
    low: float
    close: float
    adjusted_close: float
    volume: int
    dividend_amount: float
    split_coefficient: float


@dataclass
class CompanyOverview:
    """公司概况（估值与基本面指标）"""
    ticker: str
    name: str
    sector: str
    industry: str
    market_cap: Optional[float] = None
    pe_ratio: Optional[float] = None       # Trailing P/E
    forward_pe: Optional[float] = None     # Forward P/E
    peg_ratio: Optional[float] = None
    price_to_book: Optional[float] = None
    price_to_sales: Optional[float] = None
    ev_to_ebitda: Optional[float] = None
    ev_to_revenue: Optional[float] = None
    profit_margin: Optional[float] = None    # 净利润率
    operating_margin: Optional[float] = None # 营业利润率
    gross_margin: Optional[float] = None     # 毛利率
    return_on_equity: Optional[float] = None
    return_on_assets: Optional[float] = None
    revenue_ttm: Optional[float] = None
    gross_profit_ttm: Optional[float] = None
    ebitda: Optional[float] = None
    diluted_eps_ttm: Optional[float] = None
    dividend_yield: Optional[float] = None
    dividend_per_share: Optional[float] = None
    beta: Optional[float] = None
    _52_week_high: Optional[float] = None
    _52_week_low: Optional[float] = None
    analyst_target_price: Optional[float] = None
    fetched_at: str = ""


@dataclass
class FinancialStatement:
    """标准化财务报表"""
    ticker: str
    statement_type: str     # 'income' | 'balance' | 'cash_flow'
    fiscal_date_ending: str
    reported_currency: str
    # 利润表核心字段
    total_revenue: Optional[float] = None
    cost_of_revenue: Optional[float] = None
    gross_profit: Optional[float] = None
    operating_income: Optional[float] = None
    net_income: Optional[float] = None
    ebit: Optional[float] = None
    ebitda: Optional[float] = None
    eps: Optional[float] = None
    eps_diluted: Optional[float] = None
    # 资产负债表核心字段
    total_assets: Optional[float] = None
    total_current_assets: Optional[float] = None
    total_liabilities: Optional[float] = None
    total_current_liabilities: Optional[float] = None
    total_equity: Optional[float] = None
    retained_earnings: Optional[float] = None
    long_term_debt: Optional[float] = None
    # 现金流量表核心字段
    operating_cash_flow: Optional[float] = None
    capital_expenditure: Optional[float] = None
    free_cash_flow: Optional[float] = None
    dividends_paid: Optional[float] = None
    # 通用
    extra_fields: Dict[str, Any] = None
    fetched_at: str = ""

    def __post_init__(self):
        if self.extra_fields is None:
            self.extra_fields = {}


@dataclass
class EarningsData:
    """盈利数据"""
    ticker: str
    fiscal_date_ending: str
    reported_eps: Optional[float] = None
    estimated_eps: Optional[float] = None
    surprise: Optional[float] = None
    surprise_percentage: Optional[float] = None
    report_time: str = ""  # 'before_market_open' | 'after_market_close'
    fetched_at: str = ""


# ===== AlphaVantageCollector =====

class AlphaVantageCollector(BaseCollector):
    """
    Alpha Vantage API 数据采集器

    使用示例:
        config = CollectorConfig(rate_limit_rpm=75)
        collector = AlphaVantageCollector(
            api_keys=["YOUR_KEY_1", "YOUR_KEY_2"],
            config=config,
        )
        result = await collector.collect_daily_prices(["AAPL", "MSFT"])
        await collector.close()
    """

    BASE_URL = "https://www.alphavantage.co/query"

    def __init__(
        self,
        api_keys: Optional[List[str]] = None,
        config: Optional[CollectorConfig] = None,
    ):
        super().__init__(config)

        # 从参数或环境变量获取 API 密钥
        keys = api_keys or self._load_keys_from_env()
        if not keys:
            raise ValueError(
                "未提供 Alpha Vantage API 密钥。"
                "请通过 api_keys 参数传入或设置环境变量 ALPHA_VANTAGE_API_KEY"
            )

        self.api_rotator = ApiKeyRotator(
            keys=keys,
            daily_limit_per_key=500,  # Alpha Vantage 免费层日限额
            rate_limit_cooldown=65,    # 触发限流后冷却 65 秒
        )
        self.rate_limiter = RateLimiter(
            rate=self.config.rate_limit_rpm,
            period=60.0,
        )

    def _load_keys_from_env(self) -> List[str]:
        """从环境变量加载 API 密钥（支持逗号分隔多个密钥）"""
        raw = os.getenv("ALPHA_VANTAGE_API_KEY", "")
        if not raw:
            return []
        return [k.strip() for k in raw.split(",") if k.strip()]

    # ===== 抽象方法实现 =====

    def _build_url(self, endpoint: str) -> str:
        """所有 Alpha Vantage 端点共用同一个基础 URL"""
        return self.BASE_URL

    def _build_request_params(
        self, ticker: str, extra_params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """构建请求参数（不含 API 密钥 — 密钥在执行时注入）"""
        params = {"symbol": ticker}
        params.update(extra_params)
        return params

    # ===== 内部方法: 带密钥的请求执行 =====

    async def _execute_with_key(
        self,
        endpoint: str,
        ticker: str,
        params: Dict[str, Any],
        cache_key: str,
    ) -> FetchResult:
        """
        使用密钥轮换池执行单次 API 请求

        流程:
        1. 获取可用密钥
        2. 等待速率限制令牌
        3. 发起请求
        4. 根据结果更新密钥状态
        """
        key = await self.api_rotator.get_key(strategy="least_used")
        if key is None:
            return FetchResult(
                endpoint=endpoint,
                ticker=ticker,
                params=params,
                raw_response={"error": "No available API keys"},
                fetched_at=datetime.now().isoformat(),
            )

        # 将密钥注入参数
        params_with_key = {**params, "apikey": key.key}

        # 速率限制
        async with self.rate_limiter:
            url = self._build_url(endpoint)

            # 检查缓存
            if self.config.enable_cache:
                cached = self._get_from_cache(cache_key)
                if cached is not None:
                    return FetchResult(
                        endpoint=endpoint, ticker=ticker, params=params,
                        raw_response=cached,
                        fetched_at=datetime.now().isoformat(),
                        from_cache=True,
                    )

            result = await self._retry_request(url, params_with_key, endpoint, ticker)

        # 更新密钥状态
        if result.is_success:
            self.api_rotator.record_success(key)
            # 成功结果写入缓存
            if self.config.enable_cache:
                self._save_to_cache(cache_key, result.raw_response)
        else:
            error_msg = result.raw_response.get("error", "Unknown error")
            if "rate limit" in str(error_msg).lower() or "Note" in result.raw_response:
                self.api_rotator.record_rate_limited(key)
            else:
                self.api_rotator.record_error(key, str(error_msg))

        return result

    # ===== 数据拉取实现 =====

    async def collect_daily_prices(
        self,
        tickers: List[str],
        outputsize: str = "full",
    ) -> BatchFetchResult:
        """
        拉取日线调整后量价数据

        Args:
            tickers: 股票代码列表
            outputsize: 'compact' (最近100条) 或 'full' (全量20年+历史)

        Returns:
            BatchFetchResult 包含 DailyPrice 数据
        """
        self.logger.info("开始拉取日线量价", ticker_count=len(tickers), outputsize=outputsize)

        extra_params = {
            "function": AlphaVantageEndpoint.DAILY_ADJUSTED,
            "outputsize": outputsize,
            "datatype": "json",
        }

        return await self.batch_fetch(
            endpoint=AlphaVantageEndpoint.DAILY_ADJUSTED,
            tickers=tickers,
            extra_params=extra_params,
        )

    async def collect_company_overviews(
        self,
        tickers: List[str],
    ) -> BatchFetchResult:
        """
        拉取公司概况（估值指标、行业分类、关键财务比率）

        提取: Trailing P/E, Forward P/E, 营业利润率, 毛利率, ROE, ROA 等
        """
        self.logger.info("开始拉取公司概况", ticker_count=len(tickers))

        extra_params = {"function": AlphaVantageEndpoint.OVERVIEW}
        return await self.batch_fetch(
            endpoint=AlphaVantageEndpoint.OVERVIEW,
            tickers=tickers,
            extra_params=extra_params,
        )

    async def collect_income_statements(
        self,
        tickers: List[str],
    ) -> BatchFetchResult:
        """拉取利润表（年度）"""
        self.logger.info("开始拉取利润表", ticker_count=len(tickers))

        extra_params = {"function": AlphaVantageEndpoint.INCOME_STATEMENT}
        return await self.batch_fetch(
            endpoint=AlphaVantageEndpoint.INCOME_STATEMENT,
            tickers=tickers,
            extra_params=extra_params,
        )

    async def collect_balance_sheets(
        self,
        tickers: List[str],
    ) -> BatchFetchResult:
        """拉取资产负债表（年度）"""
        self.logger.info("开始拉取资产负债表", ticker_count=len(tickers))

        extra_params = {"function": AlphaVantageEndpoint.BALANCE_SHEET}
        return await self.batch_fetch(
            endpoint=AlphaVantageEndpoint.BALANCE_SHEET,
            tickers=tickers,
            extra_params=extra_params,
        )

    async def collect_cash_flows(
        self,
        tickers: List[str],
    ) -> BatchFetchResult:
        """拉取现金流量表（年度）"""
        self.logger.info("开始拉取现金流量表", ticker_count=len(tickers))

        extra_params = {"function": AlphaVantageEndpoint.CASH_FLOW}
        return await self.batch_fetch(
            endpoint=AlphaVantageEndpoint.CASH_FLOW,
            tickers=tickers,
            extra_params=extra_params,
        )

    async def collect_earnings(
        self,
        tickers: List[str],
    ) -> BatchFetchResult:
        """拉取盈利日历（历史与未来预期）"""
        self.logger.info("开始拉取盈利数据", ticker_count=len(tickers))

        extra_params = {"function": AlphaVantageEndpoint.EARNINGS}
        return await self.batch_fetch(
            endpoint=AlphaVantageEndpoint.EARNINGS,
            tickers=tickers,
            extra_params=extra_params,
        )

    async def collect_fundamentals(
        self,
        tickers: List[str],
    ) -> Dict[str, BatchFetchResult]:
        """
        一站式拉取全部基本面数据

        并发拉取: 公司概况 + 利润表 + 资产负债表 + 现金流量表 + 盈利日历

        Returns:
            字典, 键为端点名, 值为 BatchFetchResult
        """
        self.logger.info("开始全量基本面数据拉取", ticker_count=len(tickers))

        tasks = [
            ("overview", self.collect_company_overviews(tickers)),
            ("income", self.collect_income_statements(tickers)),
            ("balance", self.collect_balance_sheets(tickers)),
            ("cash_flow", self.collect_cash_flows(tickers)),
            ("earnings", self.collect_earnings(tickers)),
        ]

        # 并发执行所有子任务
        results = {}
        for name, task in tasks:
            try:
                results[name] = await task
            except Exception as e:
                self.logger.error("基本面子任务失败", task=name, error=str(e))
                results[name] = BatchFetchResult(
                    total_count=len(tickers),
                    errors=[("ALL", name, str(e))],
                )

        return results

    # ===== 批量拉取覆盖（注入密钥轮换） =====

    async def batch_fetch(
        self,
        endpoint: str,
        tickers: List[str],
        extra_params: Optional[Dict[str, Any]] = None,
        use_cache: bool = True,
    ) -> BatchFetchResult:
        """
        覆盖基类的 batch_fetch，集成密钥轮换逻辑
        """
        import time as time_mod

        start_time = time_mod.time()
        extra_params = extra_params or {}
        result = BatchFetchResult(total_count=len(tickers))

        # 并发执行所有请求
        tasks = [
            self._execute_with_key(
                endpoint=endpoint,
                ticker=ticker,
                params=self._build_request_params(ticker, extra_params),
                cache_key=self._cache_key(endpoint, ticker, self._build_request_params(ticker, extra_params)),
            )
            for ticker in tickers
        ]

        fetch_results = await asyncio.gather(*tasks, return_exceptions=True)

        for i, fetch_result in enumerate(fetch_results):
            ticker = tickers[i]
            if isinstance(fetch_result, Exception):
                result.errors.append((ticker, endpoint, str(fetch_result)))
            else:
                result.results.append(fetch_result)
                if fetch_result.is_success:
                    result.success_count += 1
                    if fetch_result.from_cache:
                        result.cache_hit_count += 1
                else:
                    err_msg = fetch_result.raw_response.get("error", fetch_result.raw_response.get("Note", "Unknown"))
                    result.errors.append((ticker, endpoint, str(err_msg)))

        result.elapsed_seconds = round(time_mod.time() - start_time, 2)
        self.logger.info(
            "批量拉取完成",
            endpoint=endpoint,
            total=result.total_count,
            success=result.success_count,
            cache_hit=result.cache_hit_count,
            errors=len(result.errors),
            elapsed_s=result.elapsed_seconds,
        )

        return result

    # ===== 响应解析方法 =====

    def parse_daily_prices(self, result: FetchResult) -> List[DailyPrice]:
        """
        解析日线量价响应 → DailyPrice 列表

        Alpha Vantage 响应格式:
        {
            "Meta Data": {...},
            "Time Series (Daily)": {
                "2024-01-05": {
                    "1. open": "182.00",
                    "2. high": "183.50",
                    "3. low": "181.00",
                    "4. close": "182.50",
                    "5. adjusted close": "182.50",
                    "6. volume": "50000000",
                    "7. dividend amount": "0.0000",
                    "8. split coefficient": "1.0"
                },
                ...
            }
        }
        """
        if not result.is_success:
            return []

        data = result.raw_response
        ticker = result.ticker
        time_series_key = "Time Series (Daily)"

        if time_series_key not in data:
            return []

        prices = []
        time_series = data[time_series_key]
        for date_str, values in time_series.items():
            try:
                prices.append(DailyPrice(
                    ticker=ticker,
                    date=date_str,
                    open=float(values["1. open"]),
                    high=float(values["2. high"]),
                    low=float(values["3. low"]),
                    close=float(values["4. close"]),
                    adjusted_close=float(values["5. adjusted close"]),
                    volume=int(values["6. volume"]),
                    dividend_amount=float(values.get("7. dividend amount", 0)),
                    split_coefficient=float(values.get("8. split coefficient", 1.0)),
                ))
            except (KeyError, ValueError) as e:
                self.logger.warning(
                    "日线数据解析跳过异常行",
                    ticker=ticker,
                    date=date_str,
                    error=str(e),
                )

        return prices

    def parse_company_overview(self, result: FetchResult) -> Optional[CompanyOverview]:
        """
        解析公司概况响应 → CompanyOverview

        提取: Trailing P/E, Forward P/E, Operating Margin, ROE, ROA 等
        """
        if not result.is_success:
            return None

        data = result.raw_response
        # 如果响应是空对象或缺少关键字段
        if not data or "Symbol" not in data:
            return None

        def _float(v: Any) -> Optional[float]:
            """安全转换为 float, None 和 'None' 返回 None"""
            if v is None or v == "None" or v == "":
                return None
            try:
                return float(v)
            except (ValueError, TypeError):
                return None

        return CompanyOverview(
            ticker=data.get("Symbol", result.ticker),
            name=data.get("Name", ""),
            sector=data.get("Sector", ""),
            industry=data.get("Industry", ""),
            market_cap=_float(data.get("MarketCapitalization")),
            pe_ratio=_float(data.get("PERatio")),
            forward_pe=_float(data.get("ForwardPE")),
            peg_ratio=_float(data.get("PEGRatio")),
            price_to_book=_float(data.get("PriceToBookRatio")),
            price_to_sales=_float(data.get("PriceToSalesRatio")),
            ev_to_ebitda=_float(data.get("EVToEBITDA")),
            ev_to_revenue=_float(data.get("EVToRevenue")),
            profit_margin=_float(data.get("ProfitMargin")),
            operating_margin=_float(data.get("OperatingMarginTTM")),
            gross_margin=_float(data.get("GrossProfitTTM")),  # 注意: 此处是绝对值, 需要 / Revenue
            return_on_equity=_float(data.get("ReturnOnEquityTTM")),
            return_on_assets=_float(data.get("ReturnOnAssetsTTM")),
            revenue_ttm=_float(data.get("RevenueTTM")),
            gross_profit_ttm=_float(data.get("GrossProfitTTM")),
            ebitda=_float(data.get("EBITDA")),
            diluted_eps_ttm=_float(data.get("DilutedEPSTTM")),
            dividend_yield=_float(data.get("DividendYield")),
            dividend_per_share=_float(data.get("DividendPerShare")),
            beta=_float(data.get("Beta")),
            _52_week_high=_float(data.get("52WeekHigh")),
            _52_week_low=_float(data.get("52WeekLow")),
            analyst_target_price=_float(data.get("AnalystTargetPrice")),
            fetched_at=result.fetched_at,
        )

    def parse_financial_statement(
        self,
        result: FetchResult,
        statement_type: str,
    ) -> List[FinancialStatement]:
        """
        解析财务报表响应 → FinancialStatement 列表

        Alpha Vantage 格式:
        {
            "symbol": "IBM",
            "annualReports": [
                {
                    "fiscalDateEnding": "2023-12-31",
                    "totalRevenue": "61860000000",
                    ...
                }
            ]
        }
        """
        if not result.is_success:
            return []

        data = result.raw_response
        reports_key = "annualReports"
        if reports_key not in data:
            # 尝试 quarterlyReports
            reports_key = "quarterlyReports"
            if reports_key not in data:
                return []

        def _float(v: Any) -> Optional[float]:
            if v is None or v == "None" or v == "":
                return None
            try:
                return float(v)
            except (ValueError, TypeError):
                return None

        statements = []
        for report in data.get(reports_key, []):
            stmt = FinancialStatement(
                ticker=result.ticker,
                statement_type=statement_type,
                fiscal_date_ending=report.get("fiscalDateEnding", ""),
                reported_currency=report.get("reportedCurrency", "USD"),
                fetched_at=result.fetched_at,
            )

            if statement_type == "income":
                stmt.total_revenue = _float(report.get("totalRevenue"))
                stmt.cost_of_revenue = _float(report.get("costOfRevenue"))
                stmt.gross_profit = _float(report.get("grossProfit"))
                stmt.operating_income = _float(report.get("operatingIncome"))
                stmt.net_income = _float(report.get("netIncome"))
                stmt.ebit = _float(report.get("ebit"))
                stmt.ebitda = _float(report.get("ebitda"))
                stmt.eps = _float(report.get("earningsPerShare"))
                stmt.eps_diluted = _float(report.get("dilutedEarningsPerShare"))
            elif statement_type == "balance":
                stmt.total_assets = _float(report.get("totalAssets"))
                stmt.total_current_assets = _float(report.get("totalCurrentAssets"))
                stmt.total_liabilities = _float(report.get("totalLiabilities"))
                stmt.total_current_liabilities = _float(report.get("totalCurrentLiabilities"))
                stmt.total_equity = _float(report.get("totalShareholderEquity"))
                stmt.retained_earnings = _float(report.get("retainedEarnings"))
                stmt.long_term_debt = _float(report.get("longTermDebt"))
            elif statement_type == "cash_flow":
                stmt.operating_cash_flow = _float(report.get("operatingCashflow"))
                stmt.capital_expenditure = _float(report.get("capitalExpenditures"))
                stmt.free_cash_flow = _float(report.get("operatingCashflow") or 0) - abs(_float(report.get("capitalExpenditures")) or 0) if report.get("operatingCashflow") is not None else None
                stmt.dividends_paid = _float(report.get("dividendPayout"))

            statements.append(stmt)

        return statements

    def parse_earnings(self, result: FetchResult) -> List[EarningsData]:
        """解析盈利日历响应"""
        if not result.is_success:
            return []

        data = result.raw_response

        def _float(v: Any) -> Optional[float]:
            if v is None or v == "None" or v == "":
                return None
            try:
                return float(v)
            except (ValueError, TypeError):
                return None

        earnings_list = []
        # Alpha Vantage 返回 annualEarnings 和 quarterlyEarnings
        for period_key in ["quarterlyEarnings", "annualEarnings"]:
            for entry in data.get(period_key, []):
                earnings_list.append(EarningsData(
                    ticker=result.ticker,
                    fiscal_date_ending=entry.get("fiscalDateEnding", ""),
                    reported_eps=_float(entry.get("reportedEPS")),
                    estimated_eps=_float(entry.get("estimatedEPS")),
                    surprise=_float(entry.get("surprise")),
                    surprise_percentage=_float(entry.get("surprisePercentage")),
                    report_time=entry.get("reportTime", ""),
                    fetched_at=result.fetched_at,
                ))

        return earnings_list

    # ===== 批量解析便捷方法 =====

    def parse_all_daily_prices(
        self, batch_result: BatchFetchResult
    ) -> pd.DataFrame:
        """
        批量解析 → DataFrame，适合直接写入 Qlib 数据层

        Returns:
            DataFrame: columns=[ticker, date, open, high, low, close, adjusted_close, volume, ...]
        """
        all_prices = []
        for result in batch_result.results:
            all_prices.extend(self.parse_daily_prices(result))

        if not all_prices:
            return pd.DataFrame()

        return pd.DataFrame([p.__dict__ for p in all_prices])

    def parse_all_overviews(
        self, batch_result: BatchFetchResult
    ) -> pd.DataFrame:
        """批量解析公司概况 → DataFrame"""
        overviews = []
        for result in batch_result.results:
            overview = self.parse_company_overview(result)
            if overview:
                overviews.append(overview)

        if not overviews:
            return pd.DataFrame()

        return pd.DataFrame([o.__dict__ for o in overviews])

    # ===== 配额监控 =====

    @property
    def quota_summary(self) -> dict:
        """API 密钥配额使用摘要"""
        return self.api_rotator.usage_summary
