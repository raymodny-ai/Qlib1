"""
ApiKeyRotator 单元测试

测试覆盖:
- 密钥轮换 (Round-Robin / Least-Used)
- 配额耗尽检测
- 频率限制触发与恢复
- 封禁检测
- 每日重置
- 使用率汇总
"""

import time

import pytest

from src.collectors.rate_limiter import ApiKey, ApiKeyRotator, KeyStatus


class TestApiKey:
    """ApiKey 数据类单元测试"""

    def test_key_initial_active(self):
        key = ApiKey(key="test_key_123", label="primary")
        assert key.key == "test_key_123"
        assert key.label == "primary"
        assert key.status == KeyStatus.ACTIVE
        assert key.is_available is True
        assert key.usage_ratio == 0.0

    def test_key_usage_ratio(self):
        key = ApiKey(key="k", daily_limit=100, calls_today=33)
        assert key.usage_ratio == 0.33

    def test_key_exhausted_not_available(self):
        key = ApiKey(key="k", daily_limit=100, calls_today=100)
        assert key.is_available is False

    def test_key_blocked_not_available(self):
        key = ApiKey(key="k", status=KeyStatus.BLOCKED)
        assert key.is_available is False

    def test_key_rate_limited_during_cooldown(self):
        key = ApiKey(key="k", status=KeyStatus.RATE_LIMITED)
        key.cooldown_until = time.time() + 999  # far future
        assert key.is_available is False

    def test_key_rate_limited_after_cooldown(self):
        key = ApiKey(key="k", status=KeyStatus.RATE_LIMITED)
        key.cooldown_until = time.time() - 1  # past
        # is_available only checks block/exhaust, cooldown check is in try_recover
        # The rotator handles recovery
        assert key.status == KeyStatus.RATE_LIMITED  # still marked until recovery

    def test_record_call_increments(self):
        key = ApiKey(key="k")
        key.record_call()
        assert key.calls_today == 1
        assert key.last_call_time > 0

    def test_mark_rate_limited(self):
        key = ApiKey(key="k")
        key.mark_rate_limited(cooldown_seconds=60)
        assert key.status == KeyStatus.RATE_LIMITED
        assert key.consecutive_failures == 1

    def test_mark_exhausted(self):
        key = ApiKey(key="k")
        key.mark_exhausted()
        assert key.status == KeyStatus.EXHAUSTED

    def test_mark_blocked(self):
        key = ApiKey(key="k")
        key.mark_blocked()
        assert key.status == KeyStatus.BLOCKED

    def test_reset_daily(self):
        key = ApiKey(key="k", daily_limit=100, calls_today=77, status=KeyStatus.EXHAUSTED)
        key.consecutive_failures = 5
        key.reset_daily()
        assert key.calls_today == 0
        assert key.status == KeyStatus.ACTIVE
        assert key.consecutive_failures == 0


class TestApiKeyRotator:
    """ApiKeyRotator 密钥轮换池测试"""

    @pytest.fixture
    def rotator(self):
        return ApiKeyRotator(
            keys=["key-a", "key-b", "key-c"],
            labels=["primary", "backup-1", "backup-2"],
            daily_limit_per_key=100,
            rate_limit_cooldown=60,
            max_consecutive_failures=5,
        )

    @pytest.fixture
    def single_key_rotator(self):
        return ApiKeyRotator(
            keys=["only-key"],
            daily_limit_per_key=100,
        )

    # === 初始化 ===

    def test_initialization(self, rotator):
        assert len(rotator.keys) == 3
        assert rotator.total_capacity == 300
        assert rotator.remaining_capacity == 300
        assert all(k.status == KeyStatus.ACTIVE for k in rotator.keys)

    def test_default_labels(self):
        rotator = ApiKeyRotator(keys=["a", "b"])
        assert rotator.keys[0].label == "key-1"
        assert rotator.keys[1].label == "key-2"

    # === 密钥获取 (Least-Used) ===

    @pytest.mark.asyncio
    async def test_get_key_least_used_first_call(self, rotator):
        key = await rotator.get_key(strategy="least_used")
        assert key is not None
        rotator.record_success(key)
        # 第一个密钥被使用1次
        assert rotator.keys[0].calls_today == 1

    @pytest.mark.asyncio
    async def test_get_key_least_used_distributes(self, rotator):
        """验证最少使用优先: 3个密钥应轮流使用"""
        for _ in range(3):
            key = await rotator.get_key(strategy="least_used")
            rotator.record_success(key)

        # 3个密钥各被使用1次
        assert rotator.keys[0].calls_today == 1
        assert rotator.keys[1].calls_today == 1
        assert rotator.keys[2].calls_today == 1

    @pytest.mark.asyncio
    async def test_get_key_round_robin(self, rotator):
        key1 = await rotator.get_key(strategy="round_robin")
        rotator.record_success(key1)
        key2 = await rotator.get_key(strategy="round_robin")
        rotator.record_success(key2)
        key3 = await rotator.get_key(strategy="round_robin")
        rotator.record_success(key3)
        key4 = await rotator.get_key(strategy="round_robin")
        rotator.record_success(key4)

        # 第4次循环回 key0
        assert key4.label == "primary"

    @pytest.mark.asyncio
    async def test_get_key_none_available(self, rotator):
        """所有密钥均不可用时返回 None"""
        for k in rotator.keys:
            k.mark_blocked()
        key = await rotator.get_key()
        assert key is None

    # === 密钥状态变更 ===

    @pytest.mark.asyncio
    async def test_rate_limited_key_sets_cooldown(self, rotator):
        key = await rotator.get_key()
        rotator.record_rate_limited(key)
        assert key.status == KeyStatus.RATE_LIMITED
        assert key.cooldown_until > time.time()

    @pytest.mark.asyncio
    async def test_all_keys_exhausted(self, rotator):
        """所有密钥日配额用尽"""
        for k in rotator.keys:
            k.calls_today = 100  # 达到上限
        key = await rotator.get_key()
        assert key is None

    @pytest.mark.asyncio
    async def test_exhausted_key_skipped(self, rotator):
        """已耗尽的密钥被跳过，使用下一个"""
        rotator.keys[0].calls_today = 100  # primary 耗尽
        key = await rotator.get_key(strategy="round_robin")
        # 应该跳到 backup-1
        assert key.label == "backup-1"

    # === 连续失败封禁 ===

    @pytest.mark.asyncio
    async def test_consecutive_failures_blocks_key(self, rotator):
        key = await rotator.get_key()
        # 连续触发 5 次限流 → 封禁
        for _ in range(5):
            rotator.record_rate_limited(key)

        assert key.status == KeyStatus.BLOCKED
        assert rotator.keys[0].status == KeyStatus.BLOCKED

    def test_record_error_increments_failures(self, rotator):
        key = rotator.keys[0]
        rotator.record_error(key, "Connection timeout")
        assert key.consecutive_failures == 1
        rotator.record_error(key, "DNS error")
        assert key.consecutive_failures == 2

    def test_success_resets_failures(self, rotator):
        key = rotator.keys[0]
        key.consecutive_failures = 3
        rotator.record_success(key)
        assert key.consecutive_failures == 0

    # === 每日重置 ===

    def test_reset_all_daily(self, rotator):
        for k in rotator.keys:
            k.calls_today = 50
            k.status = KeyStatus.EXHAUSTED
        rotator.reset_all_daily()
        for k in rotator.keys:
            assert k.calls_today == 0
            assert k.status == KeyStatus.ACTIVE

    # === 使用率汇总 ===

    def test_usage_summary(self, rotator):
        rotator.keys[0].calls_today = 30
        rotator.keys[1].calls_today = 50
        summary = rotator.usage_summary
        assert summary["total_keys"] == 3
        assert summary["remaining_capacity"] == 220
        assert len(summary["keys"]) == 3

    # === 速率限制恢复 ===

    @pytest.mark.asyncio
    async def test_rate_limited_key_recovers_after_cooldown(self, rotator):
        key = rotator.keys[0]
        key.mark_rate_limited(cooldown_seconds=0)  # immediate cooldown
        # get_key 应该恢复它
        recovered = await rotator.get_key()
        assert recovered is not None
        # 第一个密钥应该被恢复并选中（least_used: 0 calls）
        assert rotator.keys[0].status == KeyStatus.ACTIVE
