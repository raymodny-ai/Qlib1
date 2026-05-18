"""
EOD Historical Data (EODHD) API 数据采集器

继承 BaseCollector，实现 EODHD 专有的:
- 深度基本面档案 (≥30年历史)
- 企业行动 (股票拆分、股息派发)
- 宏观经济日历指标
- 与 Alpha Vantage 数据的交叉验证接口

对接端点:
- /fundamentals/{ticker}      — 深度基本面档案
- /splits/{ticker}            — 股票拆分历史
- /dividends/{ticker}         — 股息派发历史
- /macro-indicator/{country}  — 宏观经济指标
"""

import asyncio
import os
from dataclasses import dataclass, field
from datetime import date, datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd

from src.collectors.base import BaseCollector, BatchFetchResult, CollectorConfig, FetchResult
from src.collectors.rate_limiter import ApiKeyRotator, RateLimiter
from src.utils.logger import get_logger


# ===== EODHD 端点常量 =====

class EODHDEndpoint:
    """EODHD API 端点路径"""
    FUNDAMENTALS = "fundamentals"
    EOD = "eod"
    SPLITS = "splits"
    DIVIDENDS = "divs"
    MACRO_INDICATOR = "macro-indicator"
    MACRO_COUNTRY = "macro-indicators"
    BONDS_FUNDAMENTALS = "bond-fundamentals"
    BULK_FUNDAMENTALS = "bulk-fundamentals"
    CALENDAR_EARNINGS = "calendar/earnings"
    CALENDAR_TRENDS = "calendar/trends"


class MacroIndicator(Enum):
    """EODHD 宏观经济指标枚举"""
    # 美国
    USA_GDP = "USA_GDP"                         # GDP
    USA_CPI = "USA_CPI"                         # 消费者物价指数
    USA_UNEMPLOYMENT_RATE = "USA_UNEMPLOYMENT_RATE"  # 失业率
    USA_FED_FUNDS_RATE = "USA_FED_FUNDS_RATE"   # 联邦基金利率
    USA_NONFARM_PAYROLLS = "USA_NONFARM_PAYROLLS"  # 非农就业
    USA_INDUSTRIAL_PRODUCTION = "USA_INDUSTRIAL_PRODUCTION"  # 工业产出
    USA_RETAIL_SALES = "USA_RETAIL_SALES"       # 零售销售
    USA_CONSUMER_CONFIDENCE = "USA_CONSUMER_CONFIDENCE"  # 消费者信心
    USA_CORE_PCE = "USA_CORE_PCE"              # 核心PCE物价指数
    USA_PMI = "USA_PMI"                        # 采购经理人指数
    USA_ISM_MANUFACTURING = "USA_ISM_MANUFACTURING"  # ISM制造业
    USA_HOUSING_STARTS = "USA_HOUSING_STARTS"    # 新屋开工
    USA_INITIAL_JOBLESS_CLAIMS = "USA_INITIAL_JOBLESS_CLAIMS"  # 初请失业金
    USA_TRADE_BALANCE = "USA_TRADE_BALANCE"      # 贸易余额
    USA_10Y_TREASURY = "USA_10Y_TREASURY"        # 10年期国债收益率

    # 国际
    WORLD_BANK_GLOBAL_GDP = "WORLD_BANK_GLOBAL_GDP"
    IMF_GLOBAL_INFLATION = "IMF_GLOBAL_INFLATION"


class Country:
    """EODHD 国家/地区代码"""
    US = "USA"
    UK = "GBR"
    EU = "EUR"
    JP = "JPN"
    CN = "CHN"
    DE = "DEU"
    FR = "FRA"
    CA = "CAN"
    AU = "AUS"
    IN = "IND"
    BR = "BRA"
    KR = "KOR"


# ===== EODHD 专属数据模型 =====

@dataclass
class EODHDFundamentals:
    """EODHD 深度基本面档案（标准化的核心字段）"""
    ticker: str
    name: str = ""
    exchange: str = ""
    currency: str = "USD"
    sector: str = ""
    industry: str = ""
    country: str = "USA"
    isin: str = ""

    # 估值指标
    market_cap: Optional[float] = None
    enterprise_value: Optional[float] = None
    pe_ratio: Optional[float] = None        # Trailing P/E
    forward_pe: Optional[float] = None
    peg_ratio: Optional[float] = None
    price_to_book: Optional[float] = None
    price_to_sales: Optional[float] = None
    ev_to_ebitda: Optional[float] = None
    ev_to_revenue: Optional[float] = None

    # 盈利能力
    profit_margin: Optional[float] = None
    operating_margin: Optional[float] = None
    gross_margin: Optional[float] = None
    return_on_equity: Optional[float] = None
    return_on_assets: Optional[float] = None
    return_on_invested_capital: Optional[float] = None

    # 财务报表 (TTM)
    revenue_ttm: Optional[float] = None
    gross_profit_ttm: Optional[float] = None
    ebitda_ttm: Optional[float] = None
    net_income_ttm: Optional[float] = None
    free_cash_flow_ttm: Optional[float] = None
    diluted_eps_ttm: Optional[float] = None

    # 资产负债
    total_assets: Optional[float] = None
    total_debt: Optional[float] = None
    total_equity: Optional[float] = None
    current_ratio: Optional[float] = None
    debt_to_equity: Optional[float] = None

    # 股息与回购
    dividend_yield: Optional[float] = None
    dividend_per_share: Optional[float] = None
    payout_ratio: Optional[float] = None
    buyback_yield: Optional[float] = None

    # 增长率
    revenue_growth_yoy: Optional[float] = None
    earnings_growth_yoy: Optional[float] = None

    # 价格与波动
    beta: Optional[float] = None
    _52_week_high: Optional[float] = None
    _52_week_low: Optional[float] = None
    average_volume_3m: Optional[int] = None

    # 分析师
    analyst_target_price: Optional[float] = None
    number_of_analysts: Optional[int] = None

    # 完整原始数据（用于交叉验证和深度查询）
    raw_data: Dict[str, Any] = field(default_factory=dict)
    fetched_at: str = ""


@dataclass
class CorporateAction:
    """企业行动（股票拆分 / 股息派发）"""
    ticker: str
    action_type: str  # 'split' | 'dividend'
    date: str         # YYYY-MM-DD
    value: float      # 拆分系数 或 每股股息金额
    description: str = ""

    # 拆分专用
    split_from: Optional[float] = None
    split_to: Optional[float] = None

    # 股息专用
    dividend_type: Optional[str] = None  # 'cash' | 'stock'
    currency: Optional[str] = None
    declaration_date: Optional[str] = None
    record_date: Optional[str] = None
    payment_date: Optional[str] = None


@dataclass
class MacroDataPoint:
    """宏观经济数据点"""
    indicator: str         # 指标名称
    country: str           # 国家代码
    date: str              # YYYY-MM-DD
    value: float
    unit: str = ""         # 单位 (e.g., "%", "USD", "Thousands")
    frequency: str = ""    # 'monthly' | 'quarterly' | 'annual'
    description: str = ""


@dataclass
class CrossValidationResult:
    """数据交叉验证结果"""
    ticker: str
    field: str
    alpha_vantage_value: Any
    eodhd_value: Any
    deviation_pct: Optional[float] = None  # 偏差百分比
    status: str = "ok"  # 'ok' | 'warning' | 'conflict'
    resolution: str = ""  # 采用的解决策略描述
    resolved_value: Any = None  # 最终采用的值
    eodhd_priority: bool = True  # EODHD企业行动数据优先


# ===== EODHDCollector =====

class EODHDCollector(BaseCollector):
    """
    EOD Historical Data API 采集器

    特点:
    - 覆盖 ≥30 年深度基本面档案
    - 精确的企业行动数据处理（拆分/股息）
    - 宏观经济日历指标拉取
    - 与 Alpha Vantage 数据的自动交叉验证

    使用示例:
        config = CollectorConfig(rate_limit_rpm=100)
        collector = EODHDCollector(api_keys=["YOUR_EODHD_TOKEN"], config=config)
        result = await collector.collect_fundamentals(["AAPL", "MSFT"])
        await collector.close()
    """

    BASE_URL = "https://eodhistoricaldata.com/api"

    def __init__(
        self,
        api_keys: Optional[List[str]] = None,
        config: Optional[CollectorConfig] = None,
    ):
        # EODHD 速率限制较宽松，默认 100次/分钟
        if config is None:
            config = CollectorConfig(rate_limit_rpm=100)
        super().__init__(config)

        keys = api_keys or self._load_keys_from_env()
        if not keys:
            raise ValueError(
                "未提供 EODHD API 密钥。"
                "请通过 api_keys 参数传入或设置环境变量 EODHD_API_KEY"
            )

        # EODHD 通常单密钥，但仍支持多密钥轮换
        self.api_rotator = ApiKeyRotator(
            keys=keys,
            daily_limit_per_key=100000,  # EODHD 付费层日限额很高
            rate_limit_cooldown=65,
            max_consecutive_failures=10,
        )
        self.rate_limiter = RateLimiter(
            rate=self.config.rate_limit_rpm,
            period=60.0,
        )

        # Alpha Vantage 引用（用于交叉验证）
        self._av_collector = None

    def _load_keys_from_env(self) -> List[str]:
        """从环境变量加载 API Token"""
        raw = os.getenv("EODHD_API_KEY", "")
        if not raw:
            return []
        return [k.strip() for k in raw.split(",") if k.strip()]

    # ===== 抽象方法实现 =====

    def _build_url(self, endpoint: str) -> str:
        """
        EODHD 端点 URL 构建

        EODHD API 结构: {BASE_URL}/{endpoint}/{ticker}?api_token={token}
        endpoint 格式示例: "fundamentals/AAPL" (ticker 已拼入)
        """
        return f"{self.BASE_URL}/{endpoint}"

    def _build_request_params(
        self, ticker: str, extra_params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """构建请求参数（密钥在执行时注入）"""
        return extra_params

    # ===== 内部: 带密钥的请求执行 =====

    async def _fetch_with_key(
        self,
        endpoint_path: str,  # 如 "fundamentals/AAPL"
        ticker: str,
        extra_params: Dict[str, Any],
    ) -> FetchResult:
        """使用密钥轮换池执行单次 API 请求"""
        key = await self.api_rotator.get_key(strategy="least_used")
        if key is None:
            return FetchResult(
                endpoint=endpoint_path,
                ticker=ticker,
                params=extra_params,
                raw_response={"error": "No available API keys"},
                fetched_at=datetime.now().isoformat(),
            )

        # EODHD 认证: api_token 参数
        params_with_key = {**extra_params, "api_token": key.key, "fmt": "json"}

        cache_key = self._cache_key(endpoint_path, ticker, params_with_key)

        # 检查缓存
        if self.config.enable_cache:
            cached = self._get_from_cache(cache_key)
            if cached is not None:
                return FetchResult(
                    endpoint=endpoint_path, ticker=ticker, params=extra_params,
                    raw_response=cached,
                    fetched_at=datetime.now().isoformat(),
                    from_cache=True,
                )

        async with self.rate_limiter:
            url = self._build_url(endpoint_path)
            result = await self._retry_request(url, params_with_key, endpoint_path, ticker)

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

    # ===== 批量拉取覆盖 =====

    async def batch_fetch(
        self,
        endpoint: str,  # 端点模板，如 "fundamentals"
        tickers: List[str],
        extra_params: Optional[Dict[str, Any]] = None,
        use_cache: bool = True,
        endpoint_suffix: str = "",  # URL后缀
    ) -> BatchFetchResult:
        """
        覆盖基类的 batch_fetch，集成 EODHD 密钥轮换

        Args:
            endpoint: 端点名称
            tickers: 标的列表
            extra_params: 额外参数
            use_cache: 是否缓存
            endpoint_suffix: URL后缀 (如 "?filter=extended")
        """
        import time as time_mod

        start_time = time_mod.time()
        extra_params = extra_params or {}
        result = BatchFetchResult(total_count=len(tickers))

        tasks = []
        for ticker in tickers:
            endpoint_path = f"{endpoint}/{ticker}{endpoint_suffix}"
            tasks.append(self._fetch_with_key(endpoint_path, ticker, extra_params))

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
                    err_msg = fetch_result.raw_response.get("error", "Unknown error")
                    result.errors.append((ticker, endpoint, str(err_msg)))

        result.elapsed_seconds = round(time_mod.time() - start_time, 2)
        self.logger.info(
            "EODHD 批量拉取完成",
            endpoint=endpoint,
            total=result.total_count,
            success=result.success_count,
            cache_hit=result.cache_hit_count,
            errors=len(result.errors),
            elapsed_s=result.elapsed_seconds,
        )

        return result

    # ===== 数据拉取实现 =====

    async def collect_daily_prices(
        self,
        tickers: List[str],
        outputsize: str = "full",
    ) -> BatchFetchResult:
        """
        拉取日线量价数据

        EODHD /eod/{ticker} 端点返回完整历史日线数据。
        EODHD 在企业行动处理方面表现出极高精确度，
        适合作为 Alpha Vantage 数据的企业行动调整校验基准。

        Args:
            tickers: 股票代码列表
            outputsize: EODHD 默认返回全量历史
        """
        self.logger.info("EODHD 拉取日线量价", ticker_count=len(tickers))

        extra_params = {"period": "d"}  # daily
        return await self.batch_fetch(
            endpoint=EODHDEndpoint.EOD,
            tickers=tickers,
            extra_params=extra_params,
        )

    async def collect_fundamentals(
        self,
        tickers: List[str],
        filter_extended: bool = True,
    ) -> BatchFetchResult:
        """
        拉取深度基本面档案

        EODHD /fundamentals 返回极为丰富的数据:
        - General (公司基本信息)
        - Highlights (估值指标)
        - Valuation (估值比率)
        - SharesStats (股本统计)
        - Technicals (技术指标)
        - SplitsDividends (拆分与股息摘要)
        - AnalystRatings (分析师评级)
        - Earnings (盈利历史与预测)
        - Financials (财务报表: 利润表/资产负债表/现金流量表)
        - OutstandingShares (流通股历史)
        - ESG (ESG评分, 可选)

        Args:
            tickers: 股票代码列表
            filter_extended: 是否请求扩展数据
        """
        self.logger.info("EODHD 拉取深度基本面档案", ticker_count=len(tickers))

        suffix = "?filter=extended" if filter_extended else ""
        return await self.batch_fetch(
            endpoint=EODHDEndpoint.FUNDAMENTALS,
            tickers=tickers,
            endpoint_suffix=suffix,
        )

    async def collect_corporate_splits(
        self,
        tickers: List[str],
    ) -> BatchFetchResult:
        """
        拉取股票拆分历史

        EODHD 返回每个拆分事件的日期和拆分比例
        """
        self.logger.info("EODHD 拉取股票拆分历史", ticker_count=len(tickers))
        return await self.batch_fetch(
            endpoint=EODHDEndpoint.SPLITS,
            tickers=tickers,
        )

    async def collect_corporate_dividends(
        self,
        tickers: List[str],
    ) -> BatchFetchResult:
        """
        拉取股息派发历史

        返回每次派息的日期、金额、类型
        """
        self.logger.info("EODHD 拉取股息派发历史", ticker_count=len(tickers))
        return await self.batch_fetch(
            endpoint=EODHDEndpoint.DIVIDENDS,
            tickers=tickers,
        )

    async def collect_corporate_actions(
        self,
        tickers: List[str],
    ) -> Dict[str, BatchFetchResult]:
        """
        一站式拉取全部企业行动数据

        并发拉取: 拆分 + 股息
        """
        splits_result, divs_result = await asyncio.gather(
            self.collect_corporate_splits(tickers),
            self.collect_corporate_dividends(tickers),
        )
        return {
            "splits": splits_result,
            "dividends": divs_result,
        }

    async def collect_macro_indicators(
        self,
        indicators: List[str],
        country: str = "USA",
    ) -> BatchFetchResult:
        """
        拉取宏观经济指标

        Args:
            indicators: 指标列表 (如 ["USA_GDP", "USA_CPI", "USA_UNEMPLOYMENT_RATE"])
            country: 国家代码 (默认 USA)
        """
        self.logger.info("EODHD 拉取宏观经济指标", indicator_count=len(indicators), country=country)

        # 宏观经济端点使用 indicator 作为 ticker 类似标识
        return await self.batch_fetch(
            endpoint=f"{EODHDEndpoint.MACRO_INDICATOR}/{country}",
            tickers=indicators,
        )

    async def collect_all_macro(
        self,
        country: str = "USA",
        indicators: Optional[List[str]] = None,
    ) -> Dict[str, BatchFetchResult]:
        """
        一站式拉取全套宏观经济数据

        默认拉取 PRD 文档中提及的核心宏观指标
        """
        if indicators is None:
            indicators = [
                MacroIndicator.USA_GDP.value,
                MacroIndicator.USA_CPI.value,
                MacroIndicator.USA_UNEMPLOYMENT_RATE.value,
                MacroIndicator.USA_FED_FUNDS_RATE.value,
                MacroIndicator.USA_NONFARM_PAYROLLS.value,
                MacroIndicator.USA_INDUSTRIAL_PRODUCTION.value,
                MacroIndicator.USA_RETAIL_SALES.value,
                MacroIndicator.USA_CONSUMER_CONFIDENCE.value,
                MacroIndicator.USA_CORE_PCE.value,
                MacroIndicator.USA_PMI.value,
                MacroIndicator.USA_ISM_MANUFACTURING.value,
                MacroIndicator.USA_HOUSING_STARTS.value,
                MacroIndicator.USA_10Y_TREASURY.value,
            ]

        result = await self.collect_macro_indicators(indicators, country)
        self.logger.info(
            "宏观经济数据拉取完成",
            indicator_count=len(indicators),
            success=result.success_count,
            errors=len(result.errors),
        )
        return {"macro": result}

    # ===== 响应解析方法 =====

    def parse_fundamentals(self, result: FetchResult) -> Optional[EODHDFundamentals]:
        """
        解析 EODHD 基本面响应 → EODHDFundamentals

        EODHD /fundamentals 响应结构:
        {
            "General": { "Code": "AAPL", "Name": "Apple Inc.", ... },
            "Highlights": { "MarketCapitalization": 2800000000000, "PERatio": 28.5, ... },
            "Valuation": { "TrailingPE": 28.5, "ForwardPE": 26.3, ... },
            "Technicals": { "Beta": 1.25, "52WeekHigh": 199.62, ... },
            "SharesStats": { ... },
            "SplitsDividends": { ... },
            "Earnings": { ... },
            "Financials": { ... },
        }
        """
        if not result.is_success:
            return None

        data = result.raw_response
        if not data or isinstance(data, str):
            return None

        def _float(v: Any) -> Optional[float]:
            if v is None or v == "None" or v == "":
                return None
            try:
                return float(v)
            except (ValueError, TypeError):
                return None

        def _int(v: Any) -> Optional[int]:
            if v is None or v == "None" or v == "":
                return None
            try:
                return int(float(v))
            except (ValueError, TypeError):
                return None

        general = data.get("General", {})
        highlights = data.get("Highlights", {})
        valuation = data.get("Valuation", {})
        technicals = data.get("Technicals", {})
        shares = data.get("SharesStats", {})
        earnings_data = data.get("Earnings", {})

        fundamentals = EODHDFundamentals(
            ticker=data.get("Code", result.ticker),
            name=general.get("Name", general.get("CompanyName", "")),
            exchange=general.get("Exchange", ""),
            currency=general.get("CurrencyCode", "USD"),
            sector=general.get("Sector", ""),
            industry=general.get("Industry", ""),
            country=general.get("CountryName", "USA"),
            isin=general.get("ISIN", ""),

            # 估值 — 优先从 Valuation，回退到 Highlights
            market_cap=_float(highlights.get("MarketCapitalization")),
            enterprise_value=_float(valuation.get("EnterpriseValue")),
            pe_ratio=_float(valuation.get("TrailingPE") or highlights.get("PERatio")),
            forward_pe=_float(valuation.get("ForwardPE")),
            peg_ratio=_float(highlights.get("PEGRatio")),
            price_to_book=_float(valuation.get("PriceBookMRQ")),
            price_to_sales=_float(valuation.get("PriceSalesTTM")),
            ev_to_ebitda=_float(valuation.get("EVToEBITDA")),
            ev_to_revenue=_float(valuation.get("EVToRevenue")),

            # 盈利能力
            profit_margin=_float(highlights.get("ProfitMargin")),
            operating_margin=_float(highlights.get("OperatingMargin")),
            gross_margin=_float(highlights.get("GrossMargin")),
            return_on_equity=_float(highlights.get("ReturnOnEquity")),
            return_on_assets=_float(highlights.get("ReturnOnAssets")),
            return_on_invested_capital=_float(highlights.get("ReturnOnInvestedCapital")),

            # TTM 财务
            revenue_ttm=_float(highlights.get("RevenueTTM")),
            gross_profit_ttm=_float(highlights.get("GrossProfitTTM")),
            ebitda_ttm=_float(highlights.get("EBITDA")),
            net_income_ttm=_float(highlights.get("NetIncomeTTM")),
            free_cash_flow_ttm=_float(highlights.get("FreeCashFlowTTM")),
            diluted_eps_ttm=_float(highlights.get("DilutedEpsTTM")),

            # 资产负债
            total_assets=_float(highlights.get("TotalAssets")),
            total_debt=_float(highlights.get("TotalDebt")),
            total_equity=_float(highlights.get("TotalEquity")),
            current_ratio=_float(highlights.get("CurrentRatio")),
            debt_to_equity=_float(highlights.get("DebtToEquity")),

            # 股息与回购
            dividend_yield=_float(highlights.get("DividendYield")),
            dividend_per_share=_float(highlights.get("DividendShare")),
            payout_ratio=_float(highlights.get("PayoutRatio")),
            buyback_yield=_float(highlights.get("BuybackYield")),

            # 增长率
            revenue_growth_yoy=_float(highlights.get("RevenueGrowth")),
            earnings_growth_yoy=_float(highlights.get("EarningsGrowth")),

            # 价格与波动
            beta=_float(technicals.get("Beta")),
            _52_week_high=_float(technicals.get("52WeekHigh")),
            _52_week_low=_float(technicals.get("52WeekLow")),
            average_volume_3m=_int(shares.get("SharesOutstanding")),

            # 分析师
            analyst_target_price=_float(highlights.get("AnalystTargetPrice")),
            number_of_analysts=_int(highlights.get("NumberOfAnalysts")),

            raw_data=data,
            fetched_at=result.fetched_at,
        )

        return fundamentals

    def parse_splits(self, result: FetchResult) -> List[CorporateAction]:
        """
        解析股票拆分响应 → CorporateAction 列表

        EODHD /splits 返回格式:
        [{"date": "2020-08-31", "split": "4-for-1"}, ...]
        或分拆系数数值: [{"date": "2014-06-09", "split": "7.000000"}, ...]
        """
        if not result.is_success:
            return []

        data = result.raw_response
        if not isinstance(data, list):
            return []

        actions = []
        for item in data:
            date_str = item.get("date", "")
            split_value = item.get("split", "")

            # 解析拆分值: "4-for-1" → from=4, to=1; "7.000000" → from=7, to=1
            split_from = None
            split_to = None
            coefficient = None

            if isinstance(split_value, str) and "-for-" in split_value:
                parts = split_value.split("-for-")
                try:
                    split_from = float(parts[0])
                    split_to = float(parts[1])
                    coefficient = split_from / split_to if split_to != 0 else split_from
                except ValueError:
                    coefficient = float(split_value) if split_value else None
            else:
                try:
                    coefficient = float(split_value) if split_value else None
                    split_from = coefficient
                    split_to = 1.0
                except (ValueError, TypeError):
                    coefficient = None

            actions.append(CorporateAction(
                ticker=result.ticker,
                action_type="split",
                date=date_str,
                value=coefficient or 0,
                description=str(split_value),
                split_from=split_from,
                split_to=split_to,
            ))

        return actions

    def parse_dividends(self, result: FetchResult) -> List[CorporateAction]:
        """
        解析股息响应 → CorporateAction 列表

        EODHD /divs 返回格式:
        [{"date": "2024-02-09", "value": 0.24, "type": "Cash", ...}, ...]
        或 {"2024-02-09": 0.24, ...} (字典格式)
        """
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

        actions = []

        if isinstance(data, list):
            for item in data:
                actions.append(CorporateAction(
                    ticker=result.ticker,
                    action_type="dividend",
                    date=item.get("date", ""),
                    value=_float(item.get("value", 0)) or 0,
                    description=f"{item.get('type', 'Cash')} dividend",
                    dividend_type=item.get("type", "Cash"),
                    currency=item.get("currency", "USD"),
                    declaration_date=item.get("declarationDate"),
                    record_date=item.get("recordDate"),
                    payment_date=item.get("paymentDate"),
                ))
        elif isinstance(data, dict):
            # 字典格式: {date: value, date: value}
            for date_str, value in data.items():
                actions.append(CorporateAction(
                    ticker=result.ticker,
                    action_type="dividend",
                    date=date_str,
                    value=_float(value) or 0,
                    description="Cash dividend",
                    dividend_type="Cash",
                ))

        return actions

    def parse_macro_indicator(self, result: FetchResult) -> List[MacroDataPoint]:
        """
        解析宏观经济指标响应

        EODHD 返回格式:
        [{"date": "2024-01-01", "value": 4.2, "unit": "%", "frequency": "monthly", ...}, ...]
        """
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

        points = []
        records = data if isinstance(data, list) else data.get("data", [])

        for item in records:
            points.append(MacroDataPoint(
                indicator=result.ticker,  # ticker 实际存的是 indicator 名
                country=item.get("Country", "USA"),
                date=item.get("Date", item.get("date", "")),
                value=_float(item.get("Value", item.get("value", 0))) or 0,
                unit=item.get("Unit", item.get("unit", "")),
                frequency=item.get("Frequency", item.get("frequency", "")),
                description=item.get("Description", item.get("description", "")),
            ))

        return points

    # ===== 批量解析便捷方法 =====

    def parse_all_fundamentals(self, batch_result: BatchFetchResult) -> pd.DataFrame:
        """批量解析基本面 → DataFrame"""
        records = []
        for r in batch_result.results:
            f = self.parse_fundamentals(r)
            if f:
                d = {k: v for k, v in f.__dict__.items() if k != "raw_data"}
                records.append(d)

        return pd.DataFrame(records) if records else pd.DataFrame()

    def parse_all_splits(self, batch_result: BatchFetchResult) -> pd.DataFrame:
        """批量解析拆分 → DataFrame"""
        all_actions = []
        for r in batch_result.results:
            all_actions.extend(self.parse_splits(r))
        return pd.DataFrame([a.__dict__ for a in all_actions]) if all_actions else pd.DataFrame()

    def parse_all_dividends(self, batch_result: BatchFetchResult) -> pd.DataFrame:
        """批量解析股息 → DataFrame"""
        all_actions = []
        for r in batch_result.results:
            all_actions.extend(self.parse_dividends(r))
        return pd.DataFrame([a.__dict__ for a in all_actions]) if all_actions else pd.DataFrame()

    def parse_all_macro(self, batch_result: BatchFetchResult) -> pd.DataFrame:
        """批量解析宏观指标 → DataFrame"""
        all_points = []
        for r in batch_result.results:
            all_points.extend(self.parse_macro_indicator(r))
        return pd.DataFrame([p.__dict__ for p in all_points]) if all_points else pd.DataFrame()

    # ===== 交叉验证 =====

    def cross_validate_fundamentals(
        self,
        ticker: str,
        av_overview: Optional["CompanyOverview"] = None,
        av_financials: Optional[List["FinancialStatement"]] = None,
        eodhd_fundamentals: Optional[EODHDFundamentals] = None,
    ) -> List[CrossValidationResult]:
        """
        与 Alpha Vantage 数据进行交叉验证

        验证规则:
        1. 关键财务指标偏差 < 5% → ok
        2. 偏差 5-10% → warning
        3. 偏差 > 10% → conflict (以 EODHD 企业行动数据为准)

        Args:
            ticker: 股票代码
            av_overview: Alpha Vantage CompanyOverview
            av_financials: Alpha Vantage FinancialStatement 列表
            eodhd_fundamentals: EODHD 基本面数据

        Returns:
            CrossValidationResult 列表
        """
        results = []

        if not eodhd_fundamentals:
            return results

        # 可比字段映射: (field_name, alpha_vantage_source, eodhd_source)
        comparable_fields = []

        if av_overview:
            comparable_fields.extend([
                ("pe_ratio", av_overview.pe_ratio, eodhd_fundamentals.pe_ratio),
                ("forward_pe", av_overview.forward_pe, eodhd_fundamentals.forward_pe),
                ("market_cap", av_overview.market_cap, eodhd_fundamentals.market_cap),
                ("price_to_book", av_overview.price_to_book, eodhd_fundamentals.price_to_book),
                ("price_to_sales", av_overview.price_to_sales, eodhd_fundamentals.price_to_sales),
                ("ev_to_ebitda", av_overview.ev_to_ebitda, eodhd_fundamentals.ev_to_ebitda),
                ("profit_margin", av_overview.profit_margin, eodhd_fundamentals.profit_margin),
                ("operating_margin", av_overview.operating_margin, eodhd_fundamentals.operating_margin),
                ("return_on_equity", av_overview.return_on_equity, eodhd_fundamentals.return_on_equity),
                ("return_on_assets", av_overview.return_on_assets, eodhd_fundamentals.return_on_assets),
                ("beta", av_overview.beta, eodhd_fundamentals.beta),
                ("dividend_yield", av_overview.dividend_yield, eodhd_fundamentals.dividend_yield),
            ])

        # 利润表验证（取最新一期）
        if av_financials:
            latest_income = next((s for s in av_financials if s.statement_type == "income"), None)
            if latest_income:
                comparable_fields.extend([
                    ("total_revenue", latest_income.total_revenue, eodhd_fundamentals.revenue_ttm),
                    ("gross_profit", latest_income.gross_profit, eodhd_fundamentals.gross_profit_ttm),
                    ("net_income", latest_income.net_income, eodhd_fundamentals.net_income_ttm),
                ])

        for field_name, av_val, eodhd_val in comparable_fields:
            if av_val is None or eodhd_val is None:
                continue

            if av_val == 0 and eodhd_val == 0:
                deviation = 0.0
            elif av_val == 0 or eodhd_val == 0:
                deviation = 100.0  # 无法比较
            else:
                deviation = abs((av_val - eodhd_val) / av_val) * 100

            if deviation < 5:
                status = "ok"
                resolution = "两源数据一致"
            elif deviation < 10:
                status = "warning"
                resolution = f"偏差 {deviation:.1f}%，优先采用 EODHD 值"
            else:
                status = "conflict"
                resolution = f"显著偏差 {deviation:.1f}%，采用 EODHD 企业行动调整后数据"

            results.append(CrossValidationResult(
                ticker=ticker,
                field=field_name,
                alpha_vantage_value=av_val,
                eodhd_value=eodhd_val,
                deviation_pct=round(deviation, 2),
                status=status,
                resolution=resolution,
                resolved_value=eodhd_val,
                eodhd_priority=True,
            ))

        return results

    # ===== 配额监控 =====

    @property
    def quota_summary(self) -> dict:
        return self.api_rotator.usage_summary
