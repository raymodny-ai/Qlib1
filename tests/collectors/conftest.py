"""
Collectors 测试共享 fixtures

提供:
- 独立 RateLimiter 实例（每次测试新建，避免令牌状态污染）
- 独立 ApiKeyRotator 实例（每次测试新建，避免 Lock 状态污染）
- HTTP Mock 辅助工具
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest


# ===== RateLimiter 隔离 fixture =====

@pytest.fixture
def rate_limiter():
    """
    每次测试独立的 RateLimiter 实例

    避免跨测试 token 状态污染。
    """
    from src.collectors.rate_limiter import RateLimiter
    return RateLimiter(rate=75, period=60)


@pytest.fixture
def rate_limiter_fast():
    """高频率 RateLimiter（用于测试等待行为）"""
    from src.collectors.rate_limiter import RateLimiter
    return RateLimiter(rate=5, period=60)


@pytest.fixture
def rate_limiter_refill():
    """快速补充的 RateLimiter（100 tokens / 10s = 10 tokens/sec）"""
    from src.collectors.rate_limiter import RateLimiter
    return RateLimiter(rate=100, period=10)


# ===== ApiKeyRotator 隔离 fixture =====

@pytest.fixture
def api_key_rotator():
    """
    每次测试独立的 ApiKeyRotator 实例

    避免 asyncio.Lock 跨测试状态污染。
    """
    from src.collectors.rate_limiter import ApiKeyRotator
    return ApiKeyRotator(
        keys=["key-a", "key-b", "key-c"],
        labels=["primary", "backup-1", "backup-2"],
        daily_limit_per_key=100,
        rate_limit_cooldown=60,
        max_consecutive_failures=5,
    )


@pytest.fixture
def single_key_rotator():
    """单密钥 ApiKeyRotator"""
    from src.collectors.rate_limiter import ApiKeyRotator
    return ApiKeyRotator(keys=["only-key"], daily_limit_per_key=100)
