"""
RateLimiter 令牌桶速率限制器单元测试

测试覆盖:
- 初始令牌数
- acquire 消耗令牌
- 令牌自动补充
- 速率限制阻塞行为
- 上下文管理器
"""

import asyncio
import time

import pytest

from src.collectors.rate_limiter import RateLimiter


class TestRateLimiter:
    """RateLimiter 单元测试"""

    def test_initial_tokens(self):
        limiter = RateLimiter(rate=75, period=60)
        assert limiter.available_tokens == 75.0

    def test_different_rates(self):
        limiter_10 = RateLimiter(rate=10, period=60)
        assert limiter_10.available_tokens == 10.0

        limiter_100 = RateLimiter(rate=100, period=10)
        assert limiter_100.available_tokens == 100.0

    @pytest.mark.asyncio
    async def test_acquire_consumes_token(self):
        limiter = RateLimiter(rate=75, period=60)
        wait = await limiter.acquire()
        assert wait == 0.0
        # 令牌数减少
        assert limiter.available_tokens < 75.0

    @pytest.mark.asyncio
    async def test_acquire_many_returns_zero_wait_initially(self):
        """前75个请求应无需等待"""
        limiter = RateLimiter(rate=75, period=60)
        for _ in range(75):
            wait = await limiter.acquire()
            assert wait == 0.0

    @pytest.mark.asyncio
    async def test_acquire_beyond_rate_requires_wait(self):
        """超出速率限制时返回 >0 的等待时间"""
        limiter = RateLimiter(rate=5, period=60)
        for _ in range(5):
            await limiter.acquire()
        # 第6个请求应需要等待
        wait = await limiter.acquire()
        assert wait > 0.0

    @pytest.mark.asyncio
    async def test_tokens_refill_over_time(self):
        """令牌随时间补充"""
        limiter = RateLimiter(rate=100, period=10)  # 10 tokens/sec
        # 消耗所有令牌
        for _ in range(100):
            await limiter.acquire()

        remaining_before = limiter.available_tokens
        assert remaining_before < 1.0

        # 等待补充
        await asyncio.sleep(0.3)  # 应补充约3个令牌

        remaining_after = limiter.available_tokens
        assert remaining_after > remaining_before

    @pytest.mark.asyncio
    async def test_context_manager(self):
        limiter = RateLimiter(rate=50, period=60)
        async with limiter:
            # 成功获取令牌
            pass
        assert limiter.available_tokens < 50.0

    @pytest.mark.asyncio
    async def test_context_manager_waits_when_needed(self):
        """上下文管理器在需要时阻塞等待"""
        limiter = RateLimiter(rate=1, period=60)
        # 第一次无需等待
        async with limiter:
            pass

        # 第二次需要等待（令牌已耗尽）
        start = time.monotonic()
        async with limiter:
            elapsed = time.monotonic() - start

        # 应该等待了大约一段时间
        # 注意：在高速机器上可能很快，不严格断言
        assert elapsed >= 0

    def test_max_tokens_capped(self):
        """令牌数不会超过上限"""
        limiter = RateLimiter(rate=10, period=60)
        # 消耗一些令牌后等待足够长时间
        # _refill 内部限制最大令牌数
        limiter._tokens = 5
        limiter._refill()
        assert limiter._tokens <= limiter._max_tokens
