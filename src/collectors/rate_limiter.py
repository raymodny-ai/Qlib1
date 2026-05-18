"""
API 密钥轮换池与速率限制器

提供:
- ApiKeyRotator: 多密钥负载均衡、配额监控、自动故障转移
- RateLimiter: 基于滑动窗口的异步速率限制（令牌桶算法）
"""

import asyncio
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Tuple

from src.utils.logger import get_logger


# ===== 密钥轮换池 =====

class KeyStatus(Enum):
    """密钥状态枚举"""
    ACTIVE = "active"           # 正常使用中
    EXHAUSTED = "exhausted"     # 当日配额耗尽
    BLOCKED = "blocked"         # 被 API 提供商封禁
    RATE_LIMITED = "rate_limited"  # 触发频率限制


@dataclass
class ApiKey:
    """单个 API 密钥"""

    key: str
    label: str = ""  # 密钥标识（如 'primary', 'backup-1'）
    status: KeyStatus = KeyStatus.ACTIVE
    daily_limit: int = 75  # Alpha Vantage 免费层: 75次/分钟, 500次/天
    calls_today: int = 0
    last_call_time: float = 0.0
    cooldown_until: float = 0.0  # 冷却截止时间戳
    consecutive_failures: int = 0

    @property
    def is_available(self) -> bool:
        """密钥当前是否可用"""
        if self.status in (KeyStatus.BLOCKED, KeyStatus.EXHAUSTED):
            return False
        if self.status == KeyStatus.RATE_LIMITED and time.time() < self.cooldown_until:
            return False
        if self.calls_today >= self.daily_limit:
            return False
        return True

    @property
    def usage_ratio(self) -> float:
        """当日使用率"""
        return self.calls_today / self.daily_limit if self.daily_limit > 0 else 1.0

    def record_call(self) -> None:
        """记录一次调用"""
        self.calls_today += 1
        self.last_call_time = time.time()

    def mark_rate_limited(self, cooldown_seconds: int = 60) -> None:
        """标记为频率限制"""
        self.status = KeyStatus.RATE_LIMITED
        self.cooldown_until = time.time() + cooldown_seconds
        self.consecutive_failures += 1

    def mark_exhausted(self) -> None:
        """标记为配额耗尽"""
        self.status = KeyStatus.EXHAUSTED

    def mark_blocked(self) -> None:
        """标记为封禁"""
        self.status = KeyStatus.BLOCKED

    def reset_daily(self) -> None:
        """重置每日计数（应由定时任务调用）"""
        self.calls_today = 0
        if self.status == KeyStatus.EXHAUSTED:
            self.status = KeyStatus.ACTIVE
        self.consecutive_failures = 0


class ApiKeyRotator:
    """
    API 密钥轮换池

    策略:
    1. Round-Robin 轮转: 所有可用密钥轮流使用
    2. 最少使用优先: 优先选择调用次数最少的密钥
    3. 故障自动转移: 密钥被限流/封禁后自动切换到备用密钥
    4. 配额感知: 接近限额的密钥降低优先级

    使用方式:
        rotator = ApiKeyRotator(["key1", "key2", "key3"])
        key = await rotator.get_key()
        # 使用 key 发起请求...
        rotator.record_success(key)
        # 或
        rotator.record_rate_limited(key)
    """

    def __init__(
        self,
        keys: List[str],
        labels: Optional[List[str]] = None,
        daily_limit_per_key: int = 75,
        rate_limit_cooldown: int = 60,
        max_consecutive_failures: int = 10,
    ):
        self.logger = get_logger("ApiKeyRotator")
        self.rate_limit_cooldown = rate_limit_cooldown
        self.max_consecutive_failures = max_consecutive_failures

        labels = labels or [f"key-{i+1}" for i in range(len(keys))]
        self.keys: List[ApiKey] = [
            ApiKey(key=k, label=l, daily_limit=daily_limit_per_key)
            for k, l in zip(keys, labels)
        ]
        self._current_index = 0
        self._lock = asyncio.Lock()

        self.logger.info(
            "密钥轮换池初始化",
            key_count=len(self.keys),
            daily_limit_per_key=daily_limit_per_key,
        )

    @property
    def total_capacity(self) -> int:
        """总日调用容量"""
        return sum(k.daily_limit for k in self.keys)

    @property
    def remaining_capacity(self) -> int:
        """剩余日调用容量"""
        return sum(k.daily_limit - k.calls_today for k in self.keys if k.is_available)

    @property
    def usage_summary(self) -> dict:
        """使用情况汇总"""
        return {
            "total_keys": len(self.keys),
            "active_keys": sum(1 for k in self.keys if k.status == KeyStatus.ACTIVE),
            "exhausted_keys": sum(1 for k in self.keys if k.status == KeyStatus.EXHAUSTED),
            "rate_limited_keys": sum(1 for k in self.keys if k.status == KeyStatus.RATE_LIMITED),
            "blocked_keys": sum(1 for k in self.keys if k.status == KeyStatus.BLOCKED),
            "total_calls_today": sum(k.calls_today for k in self.keys),
            "remaining_capacity": self.remaining_capacity,
            "keys": [
                {
                    "label": k.label,
                    "status": k.status.value,
                    "calls_today": k.calls_today,
                    "daily_limit": k.daily_limit,
                    "usage_ratio": round(k.usage_ratio, 2),
                }
                for k in self.keys
            ],
        }

    async def get_key(self, strategy: str = "least_used") -> Optional[ApiKey]:
        """
        获取一个可用的密钥

        Args:
            strategy: 选择策略
                - 'least_used': 最少使用优先（默认）
                - 'round_robin': 轮转

        Returns:
            可用的 ApiKey 对象，无可用的返回 None
        """
        async with self._lock:
            # 首先尝试恢复 RATE_LIMITED 状态的密钥（冷却时间已过）
            now = time.time()
            for k in self.keys:
                if k.status == KeyStatus.RATE_LIMITED and now >= k.cooldown_until:
                    self.logger.info("密钥冷却完毕,恢复可用", label=k.label)
                    k.status = KeyStatus.ACTIVE

            # 筛选可用密钥
            available = [k for k in self.keys if k.is_available]

            if not available:
                self.logger.error("无可用 API 密钥", summary=self.usage_summary)
                return None

            if strategy == "round_robin":
                # Round-Robin: 从上次索引开始找
                for offset in range(len(self.keys)):
                    idx = (self._current_index + offset) % len(self.keys)
                    if self.keys[idx].is_available:
                        self._current_index = (idx + 1) % len(self.keys)
                        return self.keys[idx]
                return None

            else:  # least_used
                # 最少使用优先
                available.sort(key=lambda k: k.calls_today)
                return available[0]

    def record_success(self, key: ApiKey) -> None:
        """记录成功调用"""
        key.record_call()
        key.consecutive_failures = 0

    def record_rate_limited(self, key: ApiKey) -> None:
        """记录触发频率限制"""
        key.record_call()
        key.mark_rate_limited(self.rate_limit_cooldown)
        self.logger.warning(
            "API 密钥触发频率限制",
            label=key.label,
            cooldown_seconds=self.rate_limit_cooldown,
            consecutive_failures=key.consecutive_failures,
        )

        if key.consecutive_failures >= self.max_consecutive_failures:
            key.mark_blocked()
            self.logger.error(
                "API 密钥因连续失败被封禁",
                label=key.label,
                failures=key.consecutive_failures,
            )

    def record_error(self, key: ApiKey, error: str) -> None:
        """记录调用错误"""
        key.consecutive_failures += 1
        self.logger.error(
            "API 调用错误",
            label=key.label,
            error=error,
            consecutive_failures=key.consecutive_failures,
        )

        if key.consecutive_failures >= self.max_consecutive_failures:
            key.mark_blocked()
            self.logger.error(
                "API 密钥因连续错误被封禁",
                label=key.label,
                failures=key.consecutive_failures,
            )

    def reset_all_daily(self) -> None:
        """重置所有密钥的每日计数（UTC 0点调用）"""
        for k in self.keys:
            k.reset_daily()
        self.logger.info("所有密钥每日计数已重置")


# ===== 速率限制器（令牌桶算法） =====

class RateLimiter:
    """
    基于滑动窗口的异步令牌桶速率限制器

    防止超出 API 提供商的速率限制（如 Alpha Vantage: 75次/分钟）

    使用方式:
        limiter = RateLimiter(rate=75, period=60)
        async with limiter:
            # 执行 API 调用
            pass
    """

    def __init__(self, rate: int = 75, period: float = 60.0):
        """
        Args:
            rate: 每个 period 允许的最大调用次数
            period: 时间窗口（秒）
        """
        self.rate = rate
        self.period = period
        self._tokens = float(rate)  # 当前令牌数
        self._max_tokens = float(rate)
        self._last_refill = time.monotonic()
        self._lock = asyncio.Lock()
        self.logger = get_logger("RateLimiter")

    async def acquire(self) -> float:
        """
        获取一个令牌（阻塞直到可用）

        Returns:
            等待时间（秒），0 表示无需等待
        """
        async with self._lock:
            self._refill()

            if self._tokens >= 1.0:
                self._tokens -= 1.0
                return 0.0

            # 计算需要等待的时间
            wait_time = (1.0 - self._tokens) * (self.period / self.rate)
            self._tokens = 0.0
            return wait_time

    def _refill(self) -> None:
        """补充令牌"""
        now = time.monotonic()
        elapsed = now - self._last_refill
        new_tokens = elapsed * (self.rate / self.period)
        self._tokens = min(self._max_tokens, self._tokens + new_tokens)
        self._last_refill = now

    async def __aenter__(self):
        wait = await self.acquire()
        if wait > 0:
            await asyncio.sleep(wait)
        return self

    async def __aexit__(self, *args):
        pass

    @property
    def available_tokens(self) -> float:
        """当前可用令牌数（非阻塞检查）"""
        self._refill()
        return self._tokens
