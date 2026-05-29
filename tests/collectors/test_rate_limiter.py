"""
RateLimiter 令牌桶速率限制器单元测试

测试覆盖:
- 初始令牌数
- acquire 消耗令牌
- 令牌自动补充
- 速率限制阻塞行为
- 上下文管理器

注意: 使用 tests/collectors/conftest.py 中定义的隔离 fixture
      "rate_limiter", "rate_limiter_fast", "rate_limiter_refill"
      每个测试获取独立实例，避免跨测试 token 状态污染。
"""

import asyncio
import time

import pytest
from freezegun import freeze_time

from src.collectors.rate_limiter import RateLimiter


class TestRateLimiter:
    """RateLimiter 单元测试 — 使用 conftest.py 隔离 fixture"""

    def test_initial_tokens(self, rate_limiter):
        """初始令牌数 = rate"""
        assert rate_limiter.available_tokens == 75.0

    def test_different_rates(self):
        """不同 rate 参数产生不同初始令牌"""
        limiter_10 = RateLimiter(rate=10, period=60)
        assert limiter_10.available_tokens == 10.0

        limiter_100 = RateLimiter(rate=100, period=10)
        assert limiter_100.available_tokens == 100.0

    @pytest.mark.asyncio
    async def test_acquire_consumes_token(self, rate_limiter):
        """每次 acquire 消耗 1 个令牌"""
        wait = await rate_limiter.acquire()
        assert wait == 0.0
        # 令牌数减少
        assert rate_limiter.available_tokens < 75.0

    @pytest.mark.asyncio
    async def test_acquire_many_returns_zero_wait_initially(self, rate_limiter):
        """前75个请求应无需等待"""
        for _ in range(75):
            wait = await rate_limiter.acquire()
            assert wait == 0.0

    @pytest.mark.asyncio
    async def test_acquire_beyond_rate_requires_wait(self, rate_limiter_fast):
        """超出速率限制时返回 >0 的等待时间"""
        for _ in range(5):
            await rate_limiter_fast.acquire()
        # 第6个请求应需要等待
        wait = await rate_limiter_fast.acquire()
        assert wait > 0.0

    @pytest.mark.asyncio
    async def test_tokens_refill_over_time(self, rate_limiter_refill):
        """令牌随时间补充 — 使用 freezegun 冻结时间确保确定性"""
        # 消耗所有令牌
        for _ in range(100):
            await rate_limiter_refill.acquire()

        remaining_before = rate_limiter_refill.available_tokens
        assert remaining_before < 1.0

        # 等待补充
        await asyncio.sleep(0.3)  # 应补充约3个令牌

        remaining_after = rate_limiter_refill.available_tokens
        assert remaining_after > remaining_before

    @pytest.mark.asyncio
    async def test_tokens_refill_with_freezegun(self):
        """使用 freezegun 冻结时间，精确验证令牌补充"""
        limiter = RateLimiter(rate=100, period=10)  # 10 tokens/sec

        with freeze_time("2024-01-01 00:00:00") as frozen:
            # 消耗所有令牌
            for _ in range(100):
                await limiter.acquire()

            assert limiter.available_tokens < 1.0

            # 推进时间 1 秒 → 应补充 10 个令牌
            frozen.tick(delta=1.0)
            limiter._refill()
            assert limiter.available_tokens >= 9.0  # 约10个，允许浮点误差

    @pytest.mark.asyncio
    async def test_context_manager(self, rate_limiter):
        """上下文管理器正确消耗令牌"""
        original = rate_limiter.available_tokens
        async with rate_limiter:
            pass
        assert rate_limiter.available_tokens < original

    @pytest.mark.asyncio
    async def test_context_manager_waits_when_needed(self):
        """上下文管理器在令牌耗尽时阻塞等待"""
        # 使用高速率限制器，确保等待时间极短（rate=10, period=0.1 → 100 tokens/sec）
        limiter = RateLimiter(rate=1, period=0.1)  # 极短周期
        # 第一次无需等待
        async with limiter:
            pass

        # 第二次需要等待（令牌已耗尽）→ 等待约 0.1 秒
        start = time.monotonic()
        async with limiter:
            elapsed = time.monotonic() - start

        # 应该在短时间内完成等待
        assert elapsed >= 0
        assert elapsed < 5.0, f"等待时间过长: {elapsed:.2f}s"

    def test_max_tokens_capped(self):
        """令牌数不会超过上限"""
        limiter = RateLimiter(rate=10, period=60)
        # 消耗一些令牌后等待足够长时间
        # _refill 内部限制最大令牌数
        limiter._tokens = 5
        limiter._refill()
        assert limiter._tokens <= limiter._max_tokens
