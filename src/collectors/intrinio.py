"""
Intrinio 合规级数据采集器

继承 BaseCollector，实现 Intrinio 专有的:
- SEC 标准化 XBRL 标签基本面数据
- 期权估值指标 (Greeks / IV)
- 标准化企业财务披露数据集

对接端点:
- /companies/{ticker}                             — 公司信息
- /companies/{ticker}/fundamentals/standardized   — SEC XBRL 标准化基本面
- /options/{ticker}                               — 期权链与 Greeks
- /securities/{ticker}/prices                     — EOD 价格

Intrinio 认证: HTTP Basic Auth (API Key 为用户名, 密码留空)
"""

import asyncio
import base64
import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd

from src.collectors.base import BaseCollector, BatchFetchResult, CollectorConfig, FetchResult
from src.collectors.rate_limiter import ApiKeyRotator, RateLimiter
from src.utils.logger import get_logger


# ===== Intrinio 端点常量 =====

class IntrinioEndpoint:
    """Intrinio API 端点路径"""
    COMPANY = "companies"
    FUNDAMENTALS_STANDARDIZED = "fundamentals/standardized"
    OPTIONS = "options"
    PRICES = "prices"
    SECURITIES = "securities"


# ===== Intrinio 专属数据模型 =====

@dataclass
class IntrinioCompany:
    """Intrinio 公司基本信息"""
    ticker: str
    name: str = ""
    lei: str = ""          # Legal Entity Identifier
    cik: str = ""          # SEC CIK 编号
    sic: str = ""          # 标准行业分类
    industry: str = ""
    sector: str = ""
    employees: Optional[int] = None
    filing_frequency: str = ""  # quarterly / annual
    fetched_at: str = ""


@dataclass
class IntrinioStandardizedFundamental:
    """
    Intrinio SEC 标准化 XBRL 基本面

    来自 Intrinio 的标准化数据层，使用统一的 XBRL 标签体系，
    确保跨公司、跨时期的财务数据可比性。
    核心标签: us-gaap 命名空间 + 标准化维度。
    """
    ticker: str
    statement_type: str          # 'income_statement' / 'balance_sheet' / 'cash_flow'
    fiscal_period: str = ""      # 'FY' / 'Q1' / 'Q2' / 'Q3' / 'Q4'
    fiscal_year: int = 0
    period_end_date: str = ""    # YYYY-MM-DD
    filing_date: str = ""        # SEC 实际提交日期 (PIT关键字段)

    # 利润表 (us-gaap:IncomeStatement)
    revenue: Optional[float] = None                    # Revenues
    cost_of_revenue: Optional[float] = None            # CostOfRevenue
    gross_profit: Optional[float] = None               # GrossProfit
    operating_expenses: Optional[float] = None         # OperatingExpenses
    operating_income: Optional[float] = None           # OperatingIncomeLoss
    net_income: Optional[float] = None                 # NetIncomeLoss
    eps_basic: Optional[float] = None                  # EarningsPerShareBasic
    eps_diluted: Optional[float] = None                # EarningsPerShareDiluted
    ebit: Optional[float] = None                       # EBIT
    ebitda: Optional[float] = None                     # EBITDA
    income_tax: Optional[float] = None                 # IncomeTaxExpenseBenefit
    interest_expense: Optional[float] = None            # InterestExpense

    # 资产负债表 (us-gaap:BalanceSheet)
    total_assets: Optional[float] = None               # Assets
    current_assets: Optional[float] = None             # AssetsCurrent
    cash_and_equivalents: Optional[float] = None       # CashAndCashEquivalents
    total_liabilities: Optional[float] = None           # Liabilities
    current_liabilities: Optional[float] = None         # LiabilitiesCurrent
    long_term_debt: Optional[float] = None              # LongTermDebt
    total_equity: Optional[float] = None                # StockholdersEquity
    retained_earnings: Optional[float] = None           # RetainedEarnings
    working_capital: Optional[float] = None             # WorkingCapital

    # 现金流量表 (us-gaap:CashFlowStatement)
    operating_cash_flow: Optional[float] = None         # NetCashProvidedByUsedInOperatingActivities
    capital_expenditure: Optional[float] = None         # PaymentsToAcquirePropertyPlantAndEquipment
    free_cash_flow: Optional[float] = None              # 计算: OCF - CapEx
    financing_cash_flow: Optional[float] = None         # NetCashProvidedByUsedInFinancingActivities
    dividends_paid: Optional[float] = None               # PaymentsOfDividends

    # 全部原始标签数据
    raw_tags: Dict[str, Any] = field(default_factory=dict)
    fetched_at: str = ""


@dataclass
class IntrinioOptionMetrics:
    """期权估值指标"""
    ticker: str
    date: str              # YYYY-MM-DD
    expiration: str        # 到期日
    strike: float
    option_type: str       # 'call' / 'put'
    implied_volatility: Optional[float] = None
    delta: Optional[float] = None
    gamma: Optional[float] = None
    theta: Optional[float] = None
    vega: Optional[float] = None
    rho: Optional[float] = None
    open_interest: Optional[int] = None
    volume: Optional[int] = None
    bid: Optional[float] = None
    ask: Optional[float] = None
    last_price: Optional[float] = None
    fetched_at: str = ""


# ===== IntrinioCollector =====

class IntrinioCollector(BaseCollector):
    """
    Intrinio API 数据采集器

    特点:
    - SEC 标准化 XBRL 标签 (跨公司可比)
    - 期权 Greeks 与隐含波动率
    - 财务披露的实际提交日期 (PIT兼容)

    使用示例:
        collector = IntrinioCollector(api_keys=["YOUR_INTRINIO_KEY"])
        result = await collector.collect_standardized_fundamentals(["AAPL"], statement="income_statement")
        await collector.close()
    """

    BASE_URL = "https://api-v2.intrinio.com"

    def __init__(
        self,
        api_keys: Optional[List[str]] = None,
        config: Optional[CollectorConfig] = None,
    ):
        if config is None:
            config = CollectorConfig(rate_limit_rpm=100)
        super().__init__(config)

        keys = api_keys or self._load_keys_from_env()
        if not keys:
            raise ValueError(
                "未提供 Intrinio API 密钥。"
                "请通过 api_keys 参数传入或设置环境变量 INTRINIO_API_KEY"
            )

        self.api_rotator = ApiKeyRotator(
            keys=keys,
            daily_limit_per_key=100000,
            rate_limit_cooldown=65,
            max_consecutive_failures=10,
        )
        self.rate_limiter = RateLimiter(rate=self.config.rate_limit_rpm, period=60.0)

    def _load_keys_from_env(self) -> List[str]:
        raw = os.getenv("INTRINIO_API_KEY", "")
        return [k.strip() for k in raw.split(",") if k.strip()] if raw else []

    # ===== 抽象方法 =====

    def _build_url(self, endpoint: str) -> str:
        return f"{self.BASE_URL}/{endpoint}"

    def _build_request_params(
        self, ticker: str, extra_params: Dict[str, Any]
    ) -> Dict[str, Any]:
        return extra_params

    # ===== 认证头 (Basic Auth) =====

    def _auth_header(self, api_key: str) -> Dict[str, str]:
        """Intrinio 使用 HTTP Basic Auth: API Key 作为用户名"""
        credentials = base64.b64encode(f"{api_key}:".encode()).decode()
        return {"Authorization": f"Basic {credentials}"}

    def _default_headers(self) -> Dict[str, str]:
        return {
            "User-Agent": "Qlib-US-Fundamental/0.1.0",
            "Accept": "application/json",
            "Accept-Encoding": "gzip, deflate",
        }

    # ===== 内部: 带密钥认证的请求执行 =====

    async def _fetch_with_key(
        self, endpoint_path: str, ticker: str, extra_params: Dict[str, Any],
    ) -> FetchResult:
        key = await self.api_rotator.get_key(strategy="least_used")
        if key is None:
            return FetchResult(
                endpoint=endpoint_path, ticker=ticker, params=extra_params,
                raw_response={"error": "No available API keys"},
                fetched_at=datetime.now().isoformat(),
            )

        cache_key = self._cache_key(endpoint_path, ticker, extra_params)
        if self.config.enable_cache:
            cached = self._get_from_cache(cache_key)
            if cached is not None:
                return FetchResult(
                    endpoint=endpoint_path, ticker=ticker, params=extra_params,
                    raw_response=cached, fetched_at=datetime.now().isoformat(),
                    from_cache=True,
                )

        async with self.rate_limiter:
            url = self._build_url(endpoint_path)
            result = await self._retry_request_intrinio(url, endpoint_path, ticker, key.key, extra_params)

        if result.is_success:
            self.api_rotator.record_success(key)
            if self.config.enable_cache:
                self._save_to_cache(cache_key, result.raw_response)
        else:
            error_msg = result.raw_response.get("error", str(result.raw_response))
            if "limit" in str(error_msg).lower() or "429" in str(error_msg):
                self.api_rotator.record_rate_limited(key)
            else:
                self.api_rotator.record_error(key, str(error_msg))

        return result

    async def _retry_request_intrinio(
        self, url: str, endpoint: str, ticker: str, api_key: str, params: Dict[str, Any],
    ) -> FetchResult:
        """带 Basic Auth 头的重试请求"""
        import random as _random
        import time as _time

        headers = self._auth_header(api_key)
        last_error = None

        for attempt in range(self.config.max_retries + 1):
            try:
                session = await self._get_session()
                sem = self._semaphore
                if sem is None:
                    raise RuntimeError("Semaphore not initialized")

                async with sem:
                    async with session.get(url, params=params, headers=headers) as resp:
                        fetched_at = datetime.now().isoformat()

                        if resp.status == 429:
                            raise Exception("HTTP 429: Rate limit exceeded")
                        if resp.status != 200:
                            raise Exception(f"HTTP {resp.status}: {resp.reason}")

                        try:
                            data = await resp.json()
                        except Exception as e:
                            raise Exception(f"JSON parse error: {e}")

                        return FetchResult(
                            endpoint=endpoint, ticker=ticker, params=params,
                            raw_response=data, fetched_at=fetched_at,
                            retry_count=attempt,
                        )

            except Exception as e:
                last_error = str(e)
                if attempt < self.config.max_retries:
                    delay = min(
                        self.config.retry_base_delay * (self.config.retry_backoff_factor ** attempt),
                        self.config.retry_max_delay,
                    )
                    jitter = _random.uniform(0, delay * 0.3)
                    await asyncio.sleep(delay + jitter)

        return FetchResult(
            endpoint=endpoint, ticker=ticker, params=params,
            raw_response={"error": last_error or "Unknown"}, retry_count=self.config.max_retries,
            fetched_at=datetime.now().isoformat(),
        )

    # ===== 批量拉取 =====

    async def batch_fetch(
        self, endpoint: str, tickers: List[str],
        extra_params: Optional[Dict[str, Any]] = None,
        use_cache: bool = True,
    ) -> BatchFetchResult:
        import time as _time

        start_time = _time.time()
        extra_params = extra_params or {}
        result = BatchFetchResult(total_count=len(tickers))

        tasks = [
            self._fetch_with_key(f"{endpoint}/{ticker}", ticker, extra_params)
            for ticker in tickers
        ]
        fetch_results = await asyncio.gather(*tasks, return_exceptions=True)

        for i, fr in enumerate(fetch_results):
            ticker = tickers[i]
            if isinstance(fr, Exception):
                result.errors.append((ticker, endpoint, str(fr)))
            else:
                result.results.append(fr)
                if fr.is_success:
                    result.success_count += 1
                    if fr.from_cache:
                        result.cache_hit_count += 1
                else:
                    result.errors.append((ticker, endpoint, str(fr.raw_response)))

        result.elapsed_seconds = round(_time.time() - start_time, 2)
        self.logger.info("Intrinio 批量拉取完成", endpoint=endpoint, total=result.total_count,
                          success=result.success_count, elapsed_s=result.elapsed_seconds)
        return result

    # ===== 数据拉取实现 =====

    async def collect_daily_prices(self, tickers: List[str], outputsize: str = "full") -> BatchFetchResult:
        self.logger.info("Intrinio 拉取日线价格", ticker_count=len(tickers))
        return await self.batch_fetch("securities", tickers,
                                       extra_params={"tag": "adj_close_price", "frequency": "daily"})

    async def collect_fundamentals(self, tickers: List[str]) -> BatchFetchResult:
        self.logger.info("Intrinio 拉取公司信息", ticker_count=len(tickers))
        return await self.batch_fetch("companies", tickers)

    async def collect_standardized_fundamentals(
        self, tickers: List[str], statement: str = "income_statement",
        fiscal_year: Optional[int] = None, fiscal_period: Optional[str] = None,
    ) -> BatchFetchResult:
        """
        拉取 SEC 标准化 XBRL 基本面数据

        Args:
            tickers: 股票代码
            statement: 'income_statement' / 'balance_sheet' / 'cash_flow_statement'
            fiscal_year: 财年 (如 2023)
            fiscal_period: 'FY' / 'Q1' / 'Q2' / 'Q3' / 'Q4'
        """
        self.logger.info("Intrinio 拉取标准化基本面", ticker_count=len(tickers), statement=statement)
        params = {"statement_code": statement}
        if fiscal_year:
            params["fiscal_year"] = fiscal_year
        if fiscal_period:
            params["fiscal_period"] = fiscal_period

        return await self.batch_fetch(
            endpoint=f"companies",
            tickers=tickers,
            extra_params=params,
        )

    async def collect_options_chain(
        self, tickers: List[str], expiration: Optional[str] = None,
    ) -> BatchFetchResult:
        """
        拉取期权链数据 (Greeks + IV)

        Args:
            tickers: 股票代码
            expiration: 到期日 (YYYY-MM-DD)，None = 所有到期日
        """
        self.logger.info("Intrinio 拉取期权数据", ticker_count=len(tickers))
        params = {}
        if expiration:
            params["expiration"] = expiration
        return await self.batch_fetch("options", tickers, extra_params=params)

    # ===== 响应解析 =====

    def parse_company(self, result: FetchResult) -> Optional[IntrinioCompany]:
        if not result.is_success:
            return None
        data = result.raw_response
        if not isinstance(data, dict) or "ticker" not in data:
            return None

        return IntrinioCompany(
            ticker=data.get("ticker", result.ticker),
            name=data.get("name", ""),
            lei=data.get("lei", ""),
            cik=data.get("cik", ""),
            sic=data.get("sic", ""),
            industry=data.get("industry_category", ""),
            sector=data.get("sector", ""),
            employees=data.get("employees"),
            filing_frequency=data.get("filing_frequency", ""),
            fetched_at=result.fetched_at,
        )

    def parse_standardized_fundamental(
        self, result: FetchResult, statement_type: str = "income_statement",
    ) -> Optional[IntrinioStandardizedFundamental]:
        if not result.is_success:
            return None
        data = result.raw_response
        if not isinstance(data, dict):
            return None
        # 检测通用错误响应
        if "error" in data or "Error" in data:
            return None

        def _f(v: Any) -> Optional[float]:
            if v is None or v == "None" or v == "":
                return None
            try: return float(v)
            except (ValueError, TypeError): return None
        def _i(v: Any) -> int:
            try: return int(float(v)) if v is not None else 0
            except: return 0

        # Intrinio 返回包含 standardized_fundamentals 数组
        records = data.get("standardized_fundamentals", [data])

        if not records:
            return None

        # 取最新一条
        record = records[0] if isinstance(records, list) else records
        tags = {}

        # 构建 tag→value 映射
        for item in (records if isinstance(records, list) else [records]):
            tag = item.get("data_tag", {}).get("tag", "") if isinstance(item.get("data_tag"), dict) else item.get("tag", "")
            value = item.get("value")
            if tag:
                tags[tag] = value

        # 利润表标签映射
        if statement_type == "income_statement":
            rev = tags.get("revenues") or tags.get("Revenues")
            return IntrinioStandardizedFundamental(
                ticker=result.ticker, statement_type=statement_type,
                fiscal_period=str(record.get("fiscal_period", "")),
                fiscal_year=_i(record.get("fiscal_year", 0)),
                period_end_date=str(record.get("end_date", "")),
                filing_date=str(record.get("filing_date", "")),
                revenue=_f(rev),
                cost_of_revenue=_f(tags.get("costofrevenue") or tags.get("CostOfRevenue")),
                gross_profit=_f(tags.get("grossprofit") or tags.get("GrossProfit")),
                operating_expenses=_f(tags.get("operatingexpenses") or tags.get("OperatingExpenses")),
                operating_income=_f(tags.get("operatingincomeloss") or tags.get("OperatingIncomeLoss")),
                net_income=_f(tags.get("netincomeloss") or tags.get("NetIncomeLoss")),
                eps_basic=_f(tags.get("earningspersharebasic") or tags.get("EarningsPerShareBasic")),
                eps_diluted=_f(tags.get("earningspersharediluted")),
                ebit=_f(tags.get("ebit") or tags.get("OperatingIncomeLoss")),
                ebitda=_f(tags.get("ebitda")),
                income_tax=_f(tags.get("incometaxexpensebenefit")),
                interest_expense=_f(tags.get("interestexpense")),
                raw_tags=tags, fetched_at=result.fetched_at,
            )
        elif statement_type == "balance_sheet":
            return IntrinioStandardizedFundamental(
                ticker=result.ticker, statement_type=statement_type,
                fiscal_period=str(record.get("fiscal_period", "")),
                fiscal_year=_i(record.get("fiscal_year", 0)),
                period_end_date=str(record.get("end_date", "")),
                filing_date=str(record.get("filing_date", "")),
                total_assets=_f(tags.get("assets")),
                current_assets=_f(tags.get("assetscurrent")),
                cash_and_equivalents=_f(tags.get("cashandcashequivalentsatcarryingvalue")),
                total_liabilities=_f(tags.get("liabilities")),
                current_liabilities=_f(tags.get("liabilitiescurrent")),
                long_term_debt=_f(tags.get("longtermdebt")),
                total_equity=_f(tags.get("stockholdersequity")),
                retained_earnings=_f(tags.get("retainedearningsaccumulateddeficit")),
                working_capital=_f(tags.get("workingcapital")),
                raw_tags=tags, fetched_at=result.fetched_at,
            )
        else:  # cash_flow
            ocf = _f(tags.get("netcashprovidedbyusedinoperatingactivities"))
            capex = _f(tags.get("paymentstoacquirepropertyplantandequipment"))
            fcf = ocf + capex if ocf is not None and capex is not None else None
            return IntrinioStandardizedFundamental(
                ticker=result.ticker, statement_type=statement_type,
                fiscal_period=str(record.get("fiscal_period", "")),
                fiscal_year=_i(record.get("fiscal_year", 0)),
                period_end_date=str(record.get("end_date", "")),
                filing_date=str(record.get("filing_date", "")),
                operating_cash_flow=ocf,
                capital_expenditure=capex,
                free_cash_flow=fcf,
                financing_cash_flow=_f(tags.get("netcashprovidedbyusedinfinancingactivities")),
                dividends_paid=_f(tags.get("paymentsofdividends")),
                raw_tags=tags, fetched_at=result.fetched_at,
            )

    def parse_options(self, result: FetchResult) -> List[IntrinioOptionMetrics]:
        if not result.is_success:
            return []
        data = result.raw_response

        def _f(v): 
            return float(v) if v is not None else None
        def _i(v): 
            return int(float(v)) if v is not None else None

        options_list = data.get("options", []) if isinstance(data, dict) else []
        results = []
        for opt in options_list:
            results.append(IntrinioOptionMetrics(
                ticker=result.ticker, date=str(opt.get("date", "")),
                expiration=str(opt.get("expiration", "")),
                strike=_f(opt.get("strike", 0)) or 0,
                option_type=opt.get("type", ""),
                implied_volatility=_f(opt.get("implied_volatility")),
                delta=_f(opt.get("delta")), gamma=_f(opt.get("gamma")),
                theta=_f(opt.get("theta")), vega=_f(opt.get("vega")),
                rho=_f(opt.get("rho")),
                open_interest=_i(opt.get("open_interest")),
                volume=_i(opt.get("volume")),
                bid=_f(opt.get("bid")), ask=_f(opt.get("ask")),
                last_price=_f(opt.get("last_price")),
                fetched_at=result.fetched_at,
            ))
        return results

    # ===== 批量解析 =====

    def parse_all_companies(self, batch: BatchFetchResult) -> pd.DataFrame:
        records = []
        for r in batch.results:
            c = self.parse_company(r)
            if c: records.append({k: v for k, v in c.__dict__.items()})
        return pd.DataFrame(records) if records else pd.DataFrame()

    def parse_all_standardized(self, batch: BatchFetchResult, stmt: str = "income_statement") -> pd.DataFrame:
        records = []
        for r in batch.results:
            f = self.parse_standardized_fundamental(r, stmt)
            if f:
                d = {k: v for k, v in f.__dict__.items() if k != "raw_tags"}
                records.append(d)
        return pd.DataFrame(records) if records else pd.DataFrame()

    @property
    def quota_summary(self) -> dict:
        return self.api_rotator.usage_summary
