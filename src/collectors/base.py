"""
数据采集器抽象基类

定义所有外部数据源采集器的统一接口，包括：
- 异步 HTTP 会话管理
- 指数退避重试机制
- 标准化日志输出
- 抽象数据拉取方法

所有具体采集器（AlphaVantageCollector、EODHDCollector、IntrinioCollector）
必须继承并实现此基类定义的抽象方法。
"""

import asyncio
import hashlib
import json
import random
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import aiohttp
from aiohttp import ClientError, ClientResponse, ClientTimeout

from src.utils.logger import get_logger


# ===== 数据结构定义 =====

@dataclass
class CollectorConfig:
    """采集器通用配置"""

    # HTTP 配置
    request_timeout: int = 30  # 单次请求超时（秒）
    connect_timeout: int = 10  # 连接超时（秒）
    max_concurrent: int = 5  # 最大并发请求数

    # 重试配置
    max_retries: int = 3
    retry_base_delay: float = 1.0  # 重试基础延迟（秒）
    retry_max_delay: float = 60.0  # 重试最大延迟（秒）
    retry_backoff_factor: float = 2.0  # 指数退避因子

    # 速率限制
    rate_limit_rpm: int = 75  # 每分钟最大请求数（Alpha Vantage 免费层）

    # 缓存配置
    enable_cache: bool = True
    cache_dir: str = "./data/cache/api"
    cache_ttl_seconds: int = 3600  # 缓存有效期

    # 日志配置
    verbose: bool = False


@dataclass
class FetchResult:
    """单次 API 拉取结果"""

    endpoint: str
    ticker: str
    params: Dict[str, Any]
    raw_response: Dict[str, Any]
    fetched_at: str  # ISO 8601 时间戳
    from_cache: bool = False
    retry_count: int = 0

    @property
    def is_success(self) -> bool:
        """判断响应是否成功"""
        if not self.raw_response:
            return False
        # Alpha Vantage 错误响应特征：包含 "Error Message" 或 "Note"（频率限制）
        if "Error Message" in self.raw_response:
            return False
        if "Note" in self.raw_response and "API call frequency" in str(self.raw_response.get("Note", "")):
            return False
        return True


@dataclass
class BatchFetchResult:
    """批量拉取结果汇总"""

    results: List[FetchResult] = field(default_factory=list)
    errors: List[Tuple[str, str, str]] = field(default_factory=list)  # (ticker, endpoint, error_msg)
    total_count: int = 0
    success_count: int = 0
    cache_hit_count: int = 0
    elapsed_seconds: float = 0.0

    @property
    def success_rate(self) -> float:
        return self.success_count / self.total_count if self.total_count > 0 else 0.0


# ===== 抽象基类 =====

class BaseCollector(ABC):
    """
    数据采集器抽象基类

    提供:
    - aiohttp 会话生命周期管理
    - 指数退避重试
    - 响应缓存
    - 标准化日志

    子类必须实现:
    - _build_request_params: 构建 API 请求参数
    - _parse_response: 解析 API 响应为标准格式
    - collect_daily_prices: 拉取日线量价数据
    - collect_fundamentals: 拉取基本面数据
    """

    def __init__(self, config: Optional[CollectorConfig] = None):
        self.config = config or CollectorConfig()
        self.logger = get_logger(self.__class__.__name__)
        self._session: Optional[aiohttp.ClientSession] = None
        self._semaphore: Optional[asyncio.Semaphore] = None
        self._cache_dir: Optional[Path] = None

        # 初始化缓存目录
        if self.config.enable_cache:
            self._cache_dir = Path(self.config.cache_dir)
            self._cache_dir.mkdir(parents=True, exist_ok=True)

    # ===== 会话管理 =====

    async def _get_session(self) -> aiohttp.ClientSession:
        """获取或创建 HTTP 会话"""
        if self._session is None or self._session.closed:
            timeout = ClientTimeout(
                total=self.config.request_timeout,
                connect=self.config.connect_timeout,
            )
            self._session = aiohttp.ClientSession(
                timeout=timeout,
                headers=self._default_headers(),
            )
            self._semaphore = asyncio.Semaphore(self.config.max_concurrent)
        return self._session

    def _default_headers(self) -> Dict[str, str]:
        """默认 HTTP 请求头"""
        return {
            "User-Agent": "Qlib-US-Fundamental/0.1.0",
            "Accept": "application/json",
            "Accept-Encoding": "gzip, deflate",
            "Connection": "keep-alive",
        }

    async def close(self) -> None:
        """关闭 HTTP 会话"""
        if self._session and not self._session.closed:
            await self._session.close()
            self.logger.info("HTTP 会话已关闭")

    # ===== 缓存机制 =====

    def _cache_key(self, endpoint: str, ticker: str, params: Dict[str, Any]) -> str:
        """生成缓存键"""
        raw = f"{endpoint}:{ticker}:{json.dumps(params, sort_keys=True)}"
        return hashlib.sha256(raw.encode()).hexdigest()

    def _get_from_cache(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """从本地文件缓存读取"""
        if not self._cache_dir:
            return None

        cache_file = self._cache_dir / f"{cache_key}.json"
        if not cache_file.exists():
            return None

        # 检查缓存是否过期
        mtime = cache_file.stat().st_mtime
        age = time.time() - mtime
        if age > self.config.cache_ttl_seconds:
            return None

        try:
            with open(cache_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            self.logger.warning("缓存读取失败", cache_key=cache_key, error=str(e))
            return None

    def _save_to_cache(self, cache_key: str, data: Dict[str, Any]) -> None:
        """保存到本地文件缓存"""
        if not self._cache_dir:
            return

        cache_file = self._cache_dir / f"{cache_key}.json"
        try:
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False)
        except IOError as e:
            self.logger.warning("缓存写入失败", cache_key=cache_key, error=str(e))

    # ===== 重试机制 =====

    async def _retry_request(
        self,
        url: str,
        params: Dict[str, Any],
        endpoint: str,
        ticker: str,
    ) -> FetchResult:
        """
        带指数退避的请求重试

        Args:
            url: API 端点 URL
            params: 请求参数
            endpoint: 端点名称（用于日志）
            ticker: 股票代码

        Returns:
            FetchResult 对象
        """
        last_error: Optional[str] = None

        for attempt in range(self.config.max_retries + 1):
            try:
                session = await self._get_session()
                semaphore = self._semaphore
                if semaphore is None:
                    raise RuntimeError("Semaphore not initialized")

                async with semaphore:
                    async with session.get(url, params=params) as resp:
                        return await self._handle_response(
                            resp, endpoint, ticker, params, attempt
                        )

            except (ClientError, asyncio.TimeoutError) as e:
                last_error = f"{type(e).__name__}: {e}"
                if attempt < self.config.max_retries:
                    delay = min(
                        self.config.retry_base_delay * (self.config.retry_backoff_factor ** attempt),
                        self.config.retry_max_delay,
                    )
                    # 添加随机抖动，避免惊群效应
                    jitter = random.uniform(0, delay * 0.3)
                    total_delay = delay + jitter

                    self.logger.warning(
                        "请求失败,准备重试",
                        endpoint=endpoint,
                        ticker=ticker,
                        attempt=attempt + 1,
                        max_retries=self.config.max_retries,
                        delay_seconds=round(total_delay, 2),
                        error=last_error,
                    )
                    await asyncio.sleep(total_delay)
                else:
                    self.logger.error(
                        "请求最终失败",
                        endpoint=endpoint,
                        ticker=ticker,
                        attempts=attempt + 1,
                        error=last_error,
                    )

        # 所有重试耗尽，返回空结果
        return FetchResult(
            endpoint=endpoint,
            ticker=ticker,
            params=params,
            raw_response={"error": last_error or "Unknown error"},
            fetched_at=datetime.now(timezone.utc).isoformat(),
            retry_count=self.config.max_retries,
        )

    async def _handle_response(
        self,
        resp: ClientResponse,
        endpoint: str,
        ticker: str,
        params: Dict[str, Any],
        attempt: int,
    ) -> FetchResult:
        """处理 HTTP 响应"""
        fetched_at = datetime.now(timezone.utc).isoformat()

        if resp.status == 429:
            # 速率限制 — 抛出异常触发重试
            raise ClientError(f"HTTP 429: Rate limit exceeded for {endpoint}/{ticker}")

        if resp.status != 200:
            raise ClientError(f"HTTP {resp.status}: {resp.reason} for {endpoint}/{ticker}")

        try:
            data = await resp.json()
        except Exception as e:
            raise ClientError(f"JSON parse error for {endpoint}/{ticker}: {e}")

        if self.config.verbose:
            self.logger.debug(
                "API 请求成功",
                endpoint=endpoint,
                ticker=ticker,
                status=resp.status,
                attempt=attempt,
            )

        return FetchResult(
            endpoint=endpoint,
            ticker=ticker,
            params=params,
            raw_response=data,
            fetched_at=fetched_at,
            retry_count=attempt,
        )

    # ===== 批量拉取 =====

    async def batch_fetch(
        self,
        endpoint: str,
        tickers: List[str],
        extra_params: Optional[Dict[str, Any]] = None,
        use_cache: bool = True,
    ) -> BatchFetchResult:
        """
        并发批量拉取数据

        Args:
            endpoint: API 端点标识
            tickers: 股票代码列表
            extra_params: 额外请求参数
            use_cache: 是否使用缓存

        Returns:
            BatchFetchResult 汇总结果
        """
        start_time = time.time()
        result = BatchFetchResult(total_count=len(tickers))

        tasks = []
        for ticker in tickers:
            params = self._build_request_params(ticker, extra_params or {})
            cache_key = self._cache_key(endpoint, ticker, params)

            # 检查缓存
            if use_cache and self.config.enable_cache:
                cached = self._get_from_cache(cache_key)
                if cached is not None:
                    result.results.append(
                        FetchResult(
                            endpoint=endpoint,
                            ticker=ticker,
                            params=params,
                            raw_response=cached,
                            fetched_at=datetime.now(timezone.utc).isoformat(),
                            from_cache=True,
                        )
                    )
                    result.cache_hit_count += 1
                    result.success_count += 1
                    continue

            tasks.append(self._fetch_one(endpoint, ticker, params, cache_key))

        # 并发执行
        if tasks:
            fetch_results = await asyncio.gather(*tasks, return_exceptions=True)
            for i, fetch_result in enumerate(fetch_results):
                if isinstance(fetch_result, Exception):
                    # 找到对应的 ticker
                    ticker = tickers[result.cache_hit_count + i] if result.cache_hit_count + i < len(tickers) else "unknown"
                    result.errors.append((ticker, endpoint, str(fetch_result)))
                else:
                    result.results.append(fetch_result)
                    if fetch_result.is_success:
                        result.success_count += 1
                    else:
                        result.errors.append((fetch_result.ticker, endpoint, "API returned error response"))

        result.elapsed_seconds = round(time.time() - start_time, 2)
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

    async def _fetch_one(
        self,
        endpoint: str,
        ticker: str,
        params: Dict[str, Any],
        cache_key: str,
    ) -> FetchResult:
        """拉取单个标的的数据（含缓存写入）"""
        url = self._build_url(endpoint)
        result = await self._retry_request(url, params, endpoint, ticker)

        # 成功的结果写入缓存
        if result.is_success and self.config.enable_cache:
            self._save_to_cache(cache_key, result.raw_response)

        return result

    # ===== 子类必须实现的抽象方法 =====

    @abstractmethod
    def _build_url(self, endpoint: str) -> str:
        """
        构建 API 端点 URL

        Args:
            endpoint: 端点标识（如 'TIME_SERIES_DAILY'）

        Returns:
            完整 URL
        """
        ...

    @abstractmethod
    def _build_request_params(
        self, ticker: str, extra_params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        构建请求参数

        Args:
            ticker: 股票代码
            extra_params: 额外参数

        Returns:
            参数字典
        """
        ...

    @abstractmethod
    async def collect_daily_prices(
        self,
        tickers: List[str],
        outputsize: str = "full",
    ) -> BatchFetchResult:
        """
        拉取日线量价数据

        Args:
            tickers: 股票代码列表
            outputsize: 'compact' (最近100条) 或 'full' (全量历史)

        Returns:
            批量拉取结果
        """
        ...

    @abstractmethod
    async def collect_fundamentals(
        self,
        tickers: List[str],
    ) -> BatchFetchResult:
        """
        拉取基本面数据（公司概况 + 三张财务报表）

        Args:
            tickers: 股票代码列表

        Returns:
            批量拉取结果
        """
        ...
