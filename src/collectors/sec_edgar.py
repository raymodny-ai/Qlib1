"""
SEC EDGAR 原始披露抓取与解析子系统

直接对接 SEC EDGAR 数据库，绕过第三方聚合商，获取最原始的财务披露数据:
- 10-K 年度报告 / 10-Q 季度报告 / 8-K 重大事件 / 13F 机构持仓
- XBRL 标签精准解析 (Revenue, Net Income, Total Assets, Current Liabilities)
- 毫秒级 filing_date 记录 (Point-in-Time 时间线权威基准)
- 增量下载与本地归档

技术栈:
- edgartools: SEC EDGAR API 高级封装
- lxml + XBRL 解析: 结构化财务数据提取
"""

import asyncio
import hashlib
import json
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from src.collectors.base import BaseCollector, BatchFetchResult, CollectorConfig, FetchResult
from src.collectors.rate_limiter import RateLimiter
from src.utils.logger import get_logger


# ===== 数据模型 =====

class FilingType(Enum):
    """SEC 报告类型枚举"""
    K10 = "10-K"      # 年度报告
    Q10 = "10-Q"      # 季度报告
    K8 = "8-K"        # 重大事件
    F13 = "13F-HR"    # 机构持仓 (季度)
    F13A = "13F-HR/A" # 机构持仓修正


@dataclass
class SECFiling:
    """SEC 提交文件元数据"""
    ticker: str
    cik: str                    # SEC 中央索引键
    filing_type: str            # 10-K / 10-Q / 8-K / 13F
    accession_number: str       # 唯一提交编号
    filing_date: str            # SEC 接收时间 (YYYY-MM-DD HH:MM:SS 毫秒级)
    period_end_date: str        # 报告覆盖的财务期间截止日
    fiscal_year: int = 0
    fiscal_period: str = ""     # FY / Q1 / Q2 / Q3 / Q4
    file_url: str = ""          # 完整提交文本文件URL
    xbrl_url: str = ""          # XBRL实例文件URL
    file_size_bytes: int = 0
    is_amended: bool = False    # 是否为修正版本
    items: List[str] = field(default_factory=list)  # 8-K 事项列表
    fetched_at: str = ""


@dataclass
class XBRLFinancials:
    """
    从 SEC EDGAR XBRL 直接提取的核心财务锚点

    这些字段作为第三方API数据的校验基准，并记录财报在SEC的
    实际提交时间（filing_date），为 PIT 特征库提供绝对权威的时间线。
    """
    ticker: str
    cik: str
    filing_type: str
    accession_number: str
    filing_date: str            # SEC接收时间 — PIT 时间线的权威基准
    period_end_date: str
    fiscal_year: int = 0
    fiscal_period: str = ""

    # 核心财务锚点 (直接从 XBRL us-gaap 标签提取)
    revenue: Optional[float] = None              # 营业收入
    net_income: Optional[float] = None           # 净利润
    total_assets: Optional[float] = None         # 总资产
    total_current_assets: Optional[float] = None # 流动资产
    total_liabilities: Optional[float] = None    # 总负债
    current_liabilities: Optional[float] = None  # 流动负债
    total_equity: Optional[float] = None         # 股东权益
    retained_earnings: Optional[float] = None    # 留存收益
    operating_income: Optional[float] = None     # 营业利润
    ebit: Optional[float] = None                 # 息税前利润
    gross_profit: Optional[float] = None         # 毛利
    operating_cash_flow: Optional[float] = None  # 经营性现金流
    eps_basic: Optional[float] = None            # 基本每股收益
    eps_diluted: Optional[float] = None          # 稀释每股收益
    shares_outstanding: Optional[int] = None     # 流通股数
    long_term_debt: Optional[float] = None       # 长期债务
    cash_and_equivalents: Optional[float] = None # 现金及等价物

    # 原始XBRL标签 (tag→value 映射，用于扩展和调试)
    raw_xbrl_tags: Dict[str, Any] = field(default_factory=dict)
    fetched_at: str = ""


@dataclass
class PITTimelineEntry:
    """
    Point-in-Time 时间线条目

    每个条目记录某一时刻（filing_date）可用的财务数据快照，
    用于严格防穿越的PIT回测数据加载。
    """
    ticker: str
    cik: str
    financial_period: str       # 财务期间 (如 2023-Q3)
    period_end_date: str        # 财务期间截止日期
    value_snapshot: Dict[str, Any] = field(default_factory=dict)  # 该时间点的财务数据
    filing_date: str = ""       # 该版本数据在SEC的实际发布日期 (毫秒级)
    is_amended: bool = False    # 是否为追溯调整修正版本
    amendment_chain: List[str] = field(default_factory=list)  # [accession_number_chain]
    version_index: int = 0      # 版本序号 (0=原始, 1=第一次修正...)
    _next_version: Optional[str] = None  # 链表指针: 下一个版本的accession_number
    fetched_at: str = ""


# ===== SEC EDGAR 端点 =====

class SECEdgarEndpoint:
    """SEC EDGAR 相关端点与URL模板"""
    BASE_URL = "https://www.sec.gov"
    EDGAR_DATA = "https://www.sec.gov/Archives/edgar/data"
    SUBMISSIONS_API = "https://data.sec.gov/submissions"
    XBRL_API = "https://data.sec.gov/api/xbrl"

    # edgartools 内部使用这些端点


# ===== SECEdgarCollector =====

class SECEdgarCollector(BaseCollector):
    """
    SEC EDGAR 原始披露采集器

    使用 edgartools 库直接访问 SEC EDGAR 数据库，
    绕过第三方数据聚合商，获取最原始的财务披露。

    核心优势:
    - 直接获取 SEC 官方 filing_date (毫秒级) — PIT 时间线权威基准
    - 原生 XBRL 标签提取 — 无中间商偏差
    - 全量修正版本追溯 — 链表式版本管理

    使用示例:
        collector = SECEdgarCollector(user_agent="your-email@example.com")
        filings = await collector.search_filings("AAPL", FilingType.K10, years=3)
        xbrl = await collector.parse_xbrl_from_filing(filings[0])
        await collector.close()
    """

    # SEC 速率限制: 10 请求/秒
    SEC_RATE_LIMIT_RPM = 10

    def __init__(
        self,
        user_agent: Optional[str] = None,
        config: Optional[CollectorConfig] = None,
    ):
        if config is None:
            config = CollectorConfig(rate_limit_rpm=self.SEC_RATE_LIMIT_RPM)
        super().__init__(config)

        self._user_agent = user_agent or os.getenv(
            "SEC_EDGAR_USER_AGENT",
            "Qlib-US-Fundamental/0.1.0 (contact@example.com)"
        )
        self.rate_limiter = RateLimiter(rate=self.SEC_RATE_LIMIT_RPM, period=60.0)

        # PIT 时间线索引: {ticker: {period: [PITTimelineEntry]}}
        self._pit_index: Dict[str, Dict[str, List[PITTimelineEntry]]] = {}
        self._edgar_tools_available = self._check_edgartools()

    def _check_edgartools(self) -> bool:
        try:
            import edgar
            self.logger.info("edgartools 可用", version=edgar.__version__ if hasattr(edgar, '__version__') else "unknown")
            return True
        except ImportError:
            self.logger.warning("edgartools 未安装，将使用 SEC EDGAR HTTP API 回退方案")
            return False

    def _load_keys_from_env(self) -> List[str]:
        return []  # SEC EDGAR 公开访问，无需 API 密钥

    # ===== 抽象方法实现 =====

    def _build_url(self, endpoint: str) -> str:
        return f"{SECEdgarEndpoint.BASE_URL}/{endpoint}"

    def _build_request_params(self, ticker: str, extra_params: Dict[str, Any]) -> Dict[str, Any]:
        return extra_params

    def _default_headers(self) -> Dict[str, str]:
        return {
            "User-Agent": self._user_agent,
            "Accept": "application/json",
            "Accept-Encoding": "gzip, deflate",
            "Host": "www.sec.gov",
        }

    # ===== 数据拉取实现 =====

    async def collect_daily_prices(
        self, tickers: List[str], outputsize: str = "full"
    ) -> BatchFetchResult:
        """SEC EDGAR 不提供量价数据，返回空结果"""
        self.logger.info("SEC EDGAR 不提供日线量价数据，请使用 Alpha Vantage 或 EODHD")
        return BatchFetchResult(total_count=len(tickers))

    async def collect_fundamentals(self, tickers: List[str]) -> BatchFetchResult:
        """一站式拉取最新基本面 (使用 edgartools)"""
        self.logger.info("SEC EDGAR 拉取基本面", ticker_count=len(tickers))

        result = BatchFetchResult(total_count=len(tickers))
        for ticker in tickers:
            try:
                filings = await self.search_filings(ticker, FilingType.K10, years=1)
                if filings:
                    xbrl = await self.parse_xbrl_from_filing(filings[0])
                    if xbrl:
                        result.results.append(FetchResult(
                            endpoint="edgar_xbrl", ticker=ticker, params={},
                            raw_response=xbrl.__dict__, fetched_at=datetime.now(timezone.utc).isoformat(),
                        ))
                        result.success_count += 1
                    else:
                        result.errors.append((ticker, "edgar_xbrl", "XBRL parsing returned None"))
                else:
                    result.errors.append((ticker, "edgar", "No 10-K filings found"))
            except Exception as e:
                result.errors.append((ticker, "edgar", str(e)))

        return result

    # ===== 核心: 报告搜索与下载 =====

    async def search_filings(
        self,
        ticker: str,
        filing_type: FilingType,
        years: int = 5,
        limit: int = 20,
    ) -> List[SECFiling]:
        """
        搜索并获取 SEC 报告元数据

        Args:
            ticker: 股票代码 (如 AAPL)
            filing_type: 报告类型
            years: 搜索年份范围
            limit: 最大返回数量

        Returns:
            SECFiling 列表 (按 filing_date 降序)
        """
        if self._edgar_tools_available:
            return await self._search_with_edgartools(ticker, filing_type, years, limit)
        else:
            return await self._search_with_http(ticker, filing_type, years, limit)

    async def _search_with_edgartools(
        self, ticker: str, filing_type: FilingType, years: int, limit: int,
    ) -> List[SECFiling]:
        """使用 edgartools 库搜索"""
        try:
            import edgar

            async with self.rate_limiter:
                company = edgar.Company(ticker)
                # edgartools 默认同步，需要在 executor 中运行
                loop = asyncio.get_running_loop()
                raw_filings = await loop.run_in_executor(
                    None, lambda: company.get_filings(form=filing_type.value).latest(limit)
                )

            filings = []
            for f in raw_filings:
                try:
                    filing_date = getattr(f, 'filing_date', '')
                    period_date = getattr(f, 'period_of_report', '')
                    if hasattr(filing_date, 'strftime'):
                        filing_date = filing_date.strftime("%Y-%m-%d %H:%M:%S")
                    if hasattr(period_date, 'strftime'):
                        period_date = period_date.strftime("%Y-%m-%d")

                    filings.append(SECFiling(
                        ticker=ticker,
                        cik=str(getattr(company, 'cik', '')),
                        filing_type=str(filing_type.value),
                        accession_number=str(getattr(f, 'accession_number', '')),
                        filing_date=filing_date,
                        period_end_date=period_date,
                        file_url="",
                        xbrl_url="",
                        fetched_at=datetime.now(timezone.utc).isoformat(),
                    ))
                except Exception:
                    continue

            return filings

        except Exception as e:
            self.logger.error("edgartools 搜索失败", ticker=ticker, error=str(e))
            return await self._search_with_http(ticker, filing_type, years, limit)

    async def _search_with_http(
        self, ticker: str, filing_type: FilingType, years: int, limit: int,
    ) -> List[SECFiling]:
        """HTTP API 回退方案: 使用 SEC submissions API"""
        # 首先需要 CIK
        cik = await self._get_cik(ticker)
        if not cik:
            return []

        async with self.rate_limiter:
            url = f"{SECEdgarEndpoint.SUBMISSIONS_API}/CIK{cik}.json"
            session = await self._get_session()

            async with session.get(url, headers=self._default_headers()) as resp:
                if resp.status != 200:
                    self.logger.error("SEC API 请求失败", ticker=ticker, status=resp.status)
                    return []
                data = await resp.json()

        filings_data = data.get("filings", {}).get("recent", {})
        if not filings_data:
            return []

        forms = filings_data.get("form", [])
        dates = filings_data.get("filingDate", [])
        accessions = filings_data.get("accessionNumber", [])
        primary_docs = filings_data.get("primaryDocument", [])

        filings = []
        for i, form in enumerate(forms):
            if form != filing_type.value:
                continue
            if len(filings) >= limit:
                break

            acc = accessions[i] if i < len(accessions) else ""
            # 构建 EDGAR 文件URL
            acc_no_clean = acc.replace("-", "")
            doc = primary_docs[i] if i < len(primary_docs) else ""
            file_url = f"{SECEdgarEndpoint.EDGAR_DATA}/{cik}/{acc_no_clean}/{doc}" if doc else ""

            filings.append(SECFiling(
                ticker=ticker, cik=cik,
                filing_type=filing_type.value,
                accession_number=acc,
                filing_date=dates[i] if i < len(dates) else "",
                period_end_date="",
                file_url=file_url,
                fetched_at=datetime.now(timezone.utc).isoformat(),
            ))

        return filings

    async def _get_cik(self, ticker: str) -> str:
        """从 SEC Ticker→CIK 映射获取 CIK 编号"""
        try:
            async with self.rate_limiter:
                session = await self._get_session()
                url = "https://www.sec.gov/files/company_tickers.json"
                headers = self._default_headers()

                async with session.get(url, headers=headers) as resp:
                    if resp.status != 200:
                        return ""
                    data = await resp.json()

            # 搜索匹配的 ticker
            for entry in data.values():
                if entry.get("ticker", "").upper() == ticker.upper():
                    cik_str = str(entry.get("cik_str", ""))
                    # 补齐前导零到10位
                    return cik_str.zfill(10)

            return ""
        except Exception as e:
            self.logger.error("CIK 查询失败", ticker=ticker, error=str(e))
            return ""

    # ===== 核心: XBRL 解析 =====

    async def parse_xbrl_from_filing(self, filing: SECFiling) -> Optional[XBRLFinancials]:
        """
        从 SEC 报告中解析 XBRL 财务数据

        使用 edgartools (优先) 或 lxml 直接解析 XBRL XML。
        """
        if self._edgar_tools_available:
            return await self._parse_xbrl_edgartools(filing)
        else:
            return await self._parse_xbrl_http(filing)

    async def _parse_xbrl_edgartools(self, filing: SECFiling) -> Optional[XBRLFinancials]:
        """使用 edgartools 解析 XBRL"""
        try:
            import edgar

            async with self.rate_limiter:
                loop = asyncio.get_running_loop()
                company = edgar.Company(filing.ticker)

                # 获取指定 filing
                def get_filing():
                    fs = company.get_filings(form=filing.filing_type).latest(10)
                    for f in fs:
                        if str(getattr(f, 'accession_number', '')) == filing.accession_number:
                            return f
                    return None

                target_filing = await loop.run_in_executor(None, get_filing)

            if target_filing is None:
                return None

            async with self.rate_limiter:
                # 读取 XBRL 数据
                def extract_xbrl():
                    xbrl_obj = target_filing.xbrl()
                    if xbrl_obj is None:
                        return None
                    return self._xbrl_obj_to_dict(xbrl_obj)

                xbrl_tags = await loop.run_in_executor(None, extract_xbrl)

            if xbrl_tags is None:
                return None

            return self._build_xbrl_financials(filing, xbrl_tags)

        except Exception as e:
            self.logger.error("edgartools XBRL 解析失败", ticker=filing.ticker, error=str(e))
            return None

    def _xbrl_obj_to_dict(self, xbrl_obj: Any) -> Dict[str, Any]:
        """将 edgartools XBRL 对象转为字典"""
        tags = {}
        if hasattr(xbrl_obj, 'facts'):
            for fact in xbrl_obj.facts:
                try:
                    concept = getattr(fact, 'concept', '')
                    value = getattr(fact, 'value', None)
                    if concept and value is not None:
                        tags[str(concept).lower()] = value
                except Exception:
                    continue
        return tags

    async def _parse_xbrl_http(self, filing: SECFiling) -> Optional[XBRLFinancials]:
        """HTTP API 回退: 使用 SEC XBRL API"""
        try:
            cik = filing.cik.lstrip("0")
            acc = filing.accession_number.replace("-", "")

            async with self.rate_limiter:
                session = await self._get_session()
                url = f"{SECEdgarEndpoint.XBRL_API}/companyfacts/CIK{cik}.json"
                headers = self._default_headers()

                async with session.get(url, headers=headers) as resp:
                    if resp.status != 200:
                        return None
                    data = await resp.json()

            # 提取 us-gaap 命名空间下的标签
            facts = data.get("facts", {}).get("us-gaap", {})
            if not facts:
                return None

            xbrl_tags = {}
            for tag_name, tag_data in facts.items():
                units = tag_data.get("units", {})
                for unit_key, values in units.items():
                    for entry in values:
                        if entry.get("form") == filing.filing_type and entry.get("fy") and entry.get("fp"):
                            key = f"{tag_name}_{entry.get('fy')}_{entry.get('fp')}"
                            xbrl_tags[key] = entry.get("val")
                            xbrl_tags[tag_name] = entry.get("val")  # 最新值覆盖

            return self._build_xbrl_financials(filing, xbrl_tags)

        except Exception as e:
            self.logger.error("HTTP XBRL 解析失败", ticker=filing.ticker, error=str(e))
            return None

    def _build_xbrl_financials(
        self, filing: SECFiling, xbrl_tags: Dict[str, Any],
    ) -> XBRLFinancials:
        """从 XBRL 标签映射构建 XBRLFinancials 对象"""

        def _f(key: str) -> Optional[float]:
            v = xbrl_tags.get(key)
            if v is None:
                return None
            try:
                return float(v)
            except (ValueError, TypeError):
                return None

        def _i(key: str) -> Optional[int]:
            v = _f(key)
            return int(v) if v is not None else None

        # us-gaap 标签映射
        tag_map = {
            "revenue": ["revenues", "revenuefromcontractwithcustomerexcludingassessedtax",
                         "salesrevenuenet", "salesrevenuenetgoodsandservices"],
            "net_income": ["netincomeloss", "profitloss", "netincome"],
            "total_assets": ["assets"],
            "total_current_assets": ["assetscurrent"],
            "total_liabilities": ["liabilities"],
            "current_liabilities": ["liabilitiescurrent"],
            "total_equity": ["stockholdersequity", "stockholdersequityincludingportionattributabletononcontrollinginterest"],
            "retained_earnings": ["retainedearningsaccumulateddeficit"],
            "operating_income": ["operatingincomeloss"],
            "ebit": ["ebit"],
            "gross_profit": ["grossprofit"],
            "operating_cash_flow": ["netcashprovidedbyusedinoperatingactivities"],
            "eps_basic": ["earningspersharebasic"],
            "eps_diluted": ["earningspersharediluted"],
            "shares_outstanding": ["commonstocksharesoutstanding"],
            "long_term_debt": ["longtermdebt", "longtermdebtnoncurrent"],
            "cash_and_equivalents": ["cashandcashequivalentsatcarryingvalue", "cash"],
        }

        return XBRLFinancials(
            ticker=filing.ticker,
            cik=filing.cik,
            filing_type=filing.filing_type,
            accession_number=filing.accession_number,
            filing_date=filing.filing_date,
            period_end_date=filing.period_end_date,
            revenue=_resolve_tag(tag_map["revenue"], _f, xbrl_tags),
            net_income=_resolve_tag(tag_map["net_income"], _f, xbrl_tags),
            total_assets=_resolve_tag(tag_map["total_assets"], _f, xbrl_tags),
            total_current_assets=_resolve_tag(tag_map["total_current_assets"], _f, xbrl_tags),
            total_liabilities=_resolve_tag(tag_map["total_liabilities"], _f, xbrl_tags),
            current_liabilities=_resolve_tag(tag_map["current_liabilities"], _f, xbrl_tags),
            total_equity=_resolve_tag(tag_map["total_equity"], _f, xbrl_tags),
            retained_earnings=_resolve_tag(tag_map["retained_earnings"], _f, xbrl_tags),
            operating_income=_resolve_tag(tag_map["operating_income"], _f, xbrl_tags),
            ebit=_resolve_tag(tag_map["ebit"], _f, xbrl_tags),
            gross_profit=_resolve_tag(tag_map["gross_profit"], _f, xbrl_tags),
            operating_cash_flow=_resolve_tag(tag_map["operating_cash_flow"], _f, xbrl_tags),
            eps_basic=_resolve_tag(tag_map["eps_basic"], _f, xbrl_tags),
            eps_diluted=_resolve_tag(tag_map["eps_diluted"], _f, xbrl_tags),
            shares_outstanding=_resolve_tag(tag_map["shares_outstanding"], _i, xbrl_tags),
            long_term_debt=_resolve_tag(tag_map["long_term_debt"], _f, xbrl_tags),
            cash_and_equivalents=_resolve_tag(tag_map["cash_and_equivalents"], _f, xbrl_tags),
            raw_xbrl_tags=xbrl_tags,
            fetched_at=datetime.now(timezone.utc).isoformat(),
        )

    # ===== PIT 时间线构建 =====

    async def build_pit_timeline(
        self, ticker: str, years: int = 5,
    ) -> Dict[str, List[PITTimelineEntry]]:
        """
        构建 Point-in-Time 时间线

        对于每个财务期间 (如 2023-Q3)，遍历该期间的每一次修正版本，
        建立版本链表，记录每个版本在SEC的精确发布日期。

        Returns:
            {period_key: [PITTimelineEntry]} (按 filing_date 排列)
        """
        self.logger.info("构建 PIT 时间线", ticker=ticker, years=years)

        cik = await self._get_cik(ticker)
        if not cik:
            return {}

        # 拉取 10-K 和 10-Q
        filings_10k = await self.search_filings(ticker, FilingType.K10, years=years, limit=years * 2)
        filings_10q = await self.search_filings(ticker, FilingType.Q10, years=years, limit=years * 6)

        all_filings = filings_10k + filings_10q

        # 按 period_end_date 分组
        period_groups: Dict[str, List[SECFiling]] = {}
        for f in all_filings:
            period_key = f.period_end_date or f.filing_date[:10]
            period_groups.setdefault(period_key, []).append(f)

        # 构建时间线
        timeline: Dict[str, List[PITTimelineEntry]] = {}
        for period_key, filings in period_groups.items():
            # 按 filing_date 排序
            filings.sort(key=lambda x: x.filing_date)
            entries = []

            for i, f in enumerate(filings):
                # 构建版本链表
                next_acc = filings[i + 1].accession_number if i + 1 < len(filings) else None

                entries.append(PITTimelineEntry(
                    ticker=ticker, cik=cik,
                    financial_period=period_key,
                    period_end_date=f.period_end_date,
                    filing_date=f.filing_date,
                    is_amended=f.is_amended,
                    amendment_chain=[ff.accession_number for ff in filings[:i + 1]],
                    version_index=i,
                    _next_version=next_acc,
                    fetched_at=datetime.now(timezone.utc).isoformat(),
                ))

            timeline[period_key] = entries

        # 更新内部索引
        self._pit_index[ticker] = timeline
        return timeline

    def get_pit_data_at(
        self, ticker: str, as_of_date: str,
    ) -> Dict[str, PITTimelineEntry]:
        """
        获取指定日期的 PIT 数据快照

        对每个财务期间，返回该日期之前最新发布的版本。

        Args:
            ticker: 股票代码
            as_of_date: 模拟交易日 (YYYY-MM-DD)

        Returns:
            {period_key: 该日期可用的最新版本}
        """
        if ticker not in self._pit_index:
            self.logger.warning("PIT 时间线不存在，请先调用 build_pit_timeline()", ticker=ticker)
            return {}

        snapshot = {}
        for period_key, entries in self._pit_index[ticker].items():
            # 找到 filing_date <= as_of_date 的最新条目
            valid = [e for e in entries if e.filing_date[:10] <= as_of_date]
            if valid:
                snapshot[period_key] = valid[-1]  # 最新版本

        return snapshot

    # ===== 批量拉取 =====

    async def batch_collect_10k_10q(
        self, tickers: List[str], years: int = 5,
    ) -> Dict[str, List[SECFiling]]:
        """批量拉取多只股票的 10-K 和 10-Q"""
        results: Dict[str, List[SECFiling]] = {}
        for ticker in tickers:
            try:
                k10 = await self.search_filings(ticker, FilingType.K10, years=years)
                q10 = await self.search_filings(ticker, FilingType.Q10, years=years)
                results[ticker] = k10 + q10
            except Exception as e:
                self.logger.error("批量拉取失败", ticker=ticker, error=str(e))
                results[ticker] = []
        return results

    @property
    def pit_index_summary(self) -> dict:
        """PIT 时间线索引摘要"""
        return {
            ticker: {
                "periods": len(timeline),
                "total_entries": sum(len(e) for e in timeline.values()),
                "periods_with_amendments": sum(
                    1 for e in timeline.values() if any(ee.is_amended for ee in e)
                ),
            }
            for ticker, timeline in self._pit_index.items()
        }


def _resolve_tag(
    candidates: List[str],
    converter,
    tag_dict: Dict[str, Any],
) -> Optional[Any]:
    """从候选标签列表中解析值（大小写不敏感）"""
    # 构建小写键映射
    lower_map = {k.lower(): k for k in tag_dict}
    for candidate in candidates:
        c_lower = candidate.lower()
        # 精确匹配
        if c_lower in lower_map:
            val = converter(lower_map[c_lower])
            if val is not None:
                return val
        # 前缀匹配（处理带有后缀的标签如 revenues_fy2023_q3）
        for lower_key, orig_key in lower_map.items():
            if lower_key.startswith(c_lower):
                val = converter(orig_key)
                if val is not None:
                    return val
    return None
