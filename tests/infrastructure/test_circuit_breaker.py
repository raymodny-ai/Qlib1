"""
第4章 熔断器与自愈机制 单元测试

覆盖:
- CircuitBreaker: 状态机 / 熔断开路 / 半开恢复 / 装饰器 / 上下文管理器
- RetryPolicy: 指数退避 / 抖动 / 重试耗尽
- SlaTracker: SLA 记录 / 可用率计算 / 报告生成
- AutoHealingManager: 动作注册 / 自愈流程 / 健康检查
- with_resilience: 组合装饰器
"""

import time
import pytest

from src.infrastructure.circuit_breaker import (
    CircuitBreaker,
    CircuitState,
    CircuitBreakerOpenError,
    RetryPolicy,
    SlaTracker,
    AutoHealingManager,
    HealingAction,
    HealingResult,
    with_resilience,
)


# ============================================================================
#  CircuitBreaker 测试
# ============================================================================

class TestCircuitBreakerStates:
    """CircuitBreaker 状态机测试"""

    def test_initial_state_closed(self):
        """初始状态为 CLOSED"""
        breaker = CircuitBreaker(name="test")
        assert breaker.state == CircuitState.CLOSED
        assert not breaker.is_open

    def test_transition_to_open(self):
        """连续失败导致开路"""
        breaker = CircuitBreaker(name="test", failure_threshold=3, cooldown_seconds=0.1)

        for i in range(3):
            try:
                with breaker:
                    raise ValueError(f"fail_{i}")
            except ValueError:
                pass

        assert breaker.state == CircuitState.OPEN
        assert breaker.is_open

    def test_open_rejects_requests(self):
        """开路状态拒绝请求"""
        breaker = CircuitBreaker(name="test", failure_threshold=1, cooldown_seconds=10.0)

        # 触发开路
        try:
            with breaker:
                raise ValueError("fail")
        except ValueError:
            pass

        # 后续请求应被拒绝
        with pytest.raises(CircuitBreakerOpenError):
            with breaker:
                pass

    def test_half_open_after_cooldown(self):
        """冷却后进入 HALF_OPEN"""
        breaker = CircuitBreaker(name="test", failure_threshold=1, cooldown_seconds=0.05)

        # 触发开路
        try:
            with breaker:
                raise ValueError("fail")
        except ValueError:
            pass
        assert breaker.state == CircuitState.OPEN

        # 等待冷却
        time.sleep(0.1)

        # 下一次请求应允许通过 (进入 HALF_OPEN)
        with breaker:
            pass  # 成功！

        assert breaker.state == CircuitState.CLOSED

    def test_half_open_failure_back_to_open(self):
        """半开状态失败回到开路"""
        breaker = CircuitBreaker(name="test", failure_threshold=1, cooldown_seconds=0.05)

        # 开路
        try:
            with breaker:
                raise ValueError("fail1")
        except ValueError:
            pass
        time.sleep(0.1)

        # 半开 → 失败 → 回到开路
        try:
            with breaker:
                raise ValueError("fail2")
        except ValueError:
            pass

        assert breaker.state == CircuitState.OPEN

    def test_success_resets_failure_count(self):
        """成功重置失败计数"""
        breaker = CircuitBreaker(name="test", failure_threshold=5)

        try:
            with breaker:
                raise ValueError("e1")
        except ValueError:
            pass
        try:
            with breaker:
                raise ValueError("e2")
        except ValueError:
            pass

        # 成功
        with breaker:
            pass

        # 失败计数归零
        assert breaker._failure_count == 0

    def test_call_method(self):
        """call 方法显式调用"""
        breaker = CircuitBreaker(name="test")

        def add(a, b):
            return a + b

        result = breaker.call(add, 1, 2)
        assert result == 3

    def test_call_method_failure(self):
        """call 方法失败传播"""
        breaker = CircuitBreaker(name="test", failure_threshold=3)

        def fail():
            raise RuntimeError("boom")

        with pytest.raises(RuntimeError):
            breaker.call(fail)

    def test_decorator_mode(self):
        """装饰器模式"""
        breaker = CircuitBreaker(name="deco_test")

        @breaker
        def safe_add(a, b):
            return a + b

        result = safe_add(3, 4)
        assert result == 7

    def test_decorator_with_failure(self):
        """装饰器失败触发熔断"""
        breaker = CircuitBreaker(name="deco_fail", failure_threshold=2)

        call_count = [0]

        @breaker
        def flaky():
            call_count[0] += 1
            if call_count[0] <= 2:
                raise ValueError("flaky")
            return "ok"

        # 前两次失败 → 开路
        for _ in range(2):
            try:
                flaky()
            except ValueError:
                pass

        # 第三次被拒绝
        with pytest.raises(CircuitBreakerOpenError):
            flaky()

    def test_exception_type_filter(self):
        """异常类型过滤"""
        breaker = CircuitBreaker(
            name="filter_test",
            failure_threshold=2,
            exception_types=(ValueError,),
        )

        # RuntimeError 不计入熔断
        try:
            with breaker:
                raise RuntimeError("not counted")
        except RuntimeError:
            pass

        assert breaker._failure_count == 0
        assert breaker.state == CircuitState.CLOSED

        # ValueError 计入熔断
        try:
            with breaker:
                raise ValueError("counted")
        except ValueError:
            pass

        assert breaker._failure_count == 1

    def test_callbacks(self):
        """状态转换回调"""
        open_calls = []
        close_calls = []

        breaker = CircuitBreaker(
            name="cb_test",
            failure_threshold=1,
            cooldown_seconds=0.05,
            on_open=lambda name: open_calls.append(name),
            on_close=lambda name: close_calls.append(name),
        )

        # 触发开路
        try:
            with breaker:
                raise ValueError("fail")
        except ValueError:
            pass

        assert len(open_calls) == 1
        assert open_calls[0] == "cb_test"

        # 等待冷却 → 半开 → 成功 → 闭合
        time.sleep(0.1)
        with breaker:
            pass

        assert len(close_calls) == 1
        assert close_calls[0] == "cb_test"

    def test_stats(self):
        """统计信息"""
        breaker = CircuitBreaker(name="stats_test", failure_threshold=5)

        with breaker:
            pass
        with breaker:
            pass
        try:
            with breaker:
                raise ValueError("e")
        except ValueError:
            pass

        stats = breaker.get_stats()
        assert stats["name"] == "stats_test"
        assert stats["total_requests"] == 3
        assert stats["total_successes"] == 2
        assert stats["total_failures"] == 1
        assert stats["state"] == "closed"

    def test_health_score(self):
        """健康分数"""
        breaker = CircuitBreaker(name="health_test", failure_threshold=5)

        with breaker:
            pass
        with breaker:
            pass
        try:
            with breaker:
                raise ValueError("e")
        except ValueError:
            pass

        assert breaker.health_score == pytest.approx(66.7, rel=0.1)

    def test_reset(self):
        """手动重置"""
        breaker = CircuitBreaker(name="reset_test", failure_threshold=2)

        for _ in range(2):
            try:
                with breaker:
                    raise ValueError("fail")
            except ValueError:
                pass

        assert breaker.state == CircuitState.OPEN

        breaker.reset()
        assert breaker.state == CircuitState.CLOSED
        assert breaker._failure_count == 0

    def test_half_open_max_requests(self):
        """半开状态试探请求限制：成功后转 CLOSED，不再限制"""
        breaker = CircuitBreaker(
            name="ho_test",
            failure_threshold=1,
            cooldown_seconds=0.05,
            half_open_max_requests=1,
        )

        # 开路
        try:
            with breaker:
                raise ValueError("fail")
        except ValueError:
            pass
        time.sleep(0.1)

        # 第一次半开请求通过 → 转为 CLOSED
        with breaker:
            pass

        assert breaker.state == CircuitState.CLOSED

        # CLOSED 状态下所有请求正常通过
        with breaker:
            pass


# ============================================================================
#  RetryPolicy 测试
# ============================================================================

class TestRetryPolicy:
    """RetryPolicy 重试策略测试"""

    def test_no_retry_on_success(self):
        """成功时无重试"""
        retry = RetryPolicy(max_retries=3)
        call_count = [0]

        @retry
        def ok():
            call_count[0] += 1
            return "done"

        result = ok()
        assert result == "done"
        assert call_count[0] == 1

    def test_retry_on_failure(self):
        """失败时重试"""
        retry = RetryPolicy(max_retries=3, initial_delay=0.01)
        call_count = [0]

        @retry
        def flaky():
            call_count[0] += 1
            if call_count[0] < 3:
                raise ValueError("not yet")
            return "finally"

        result = flaky()
        assert result == "finally"
        assert call_count[0] == 3

    def test_retry_exhausted(self):
        """重试耗尽"""
        retry = RetryPolicy(max_retries=2, initial_delay=0.01)
        call_count = [0]

        @retry
        def always_fails():
            call_count[0] += 1
            raise RuntimeError("always")

        with pytest.raises(RuntimeError):
            always_fails()

        # 1 次初始 + 2 次重试 = 3 次
        assert call_count[0] == 3

    def test_exception_type_filter(self):
        """可重试异常过滤"""
        retry = RetryPolicy(max_retries=2, exception_types=(ValueError,), initial_delay=0.01)

        call_count = [0]

        @retry
        def raises_type_error():
            call_count[0] += 1
            raise TypeError("not retried")

        with pytest.raises(TypeError):
            raises_type_error()

        assert call_count[0] == 1  # 未重试

    def test_compute_delay(self):
        """延迟计算"""
        retry = RetryPolicy(
            max_retries=5,
            backoff_base=2.0,
            initial_delay=1.0,
            max_delay=60.0,
            jitter=False,
        )
        # attempt 1: 1.0 * 2^0 = 1.0
        assert retry.compute_delay(1) == 1.0
        # attempt 3: 1.0 * 2^2 = 4.0
        assert retry.compute_delay(3) == 4.0

    def test_max_delay_cap(self):
        """最大延迟上限"""
        retry = RetryPolicy(
            max_retries=10,
            backoff_base=2.0,
            initial_delay=1.0,
            max_delay=5.0,
            jitter=False,
        )
        # 2^10 = 1024 >> 5.0
        delay = retry.compute_delay(11)
        assert delay <= 5.0

    def test_jitter(self):
        """随机抖动"""
        retry = RetryPolicy(
            initial_delay=10.0,
            jitter=True,
        )
        delays = [retry.compute_delay(1) for _ in range(50)]
        # 所有值应在 7.5 ~ 12.5 范围内 (±25%)
        for d in delays:
            assert 7.0 <= d <= 13.0

    def test_call_method(self):
        """显式 call 方法"""
        retry = RetryPolicy(max_retries=2, initial_delay=0.01)
        call_count = [0]

        def flaky():
            call_count[0] += 1
            if call_count[0] < 2:
                raise ValueError("nope")
            return "ok"

        result = retry.call(flaky)
        assert result == "ok"

    def test_stats(self):
        """统计信息"""
        retry = RetryPolicy(max_retries=3, initial_delay=0.01)
        count = [0]

        def flaky():
            count[0] += 1
            if count[0] < 3:
                raise ValueError("nope")
            return "ok"

        retry.call(flaky)

        stats = retry.get_stats()
        assert stats["total_attempts"] == 3
        assert stats["total_retries"] == 2

    def test_on_retry_callback(self):
        """重试回调"""
        callback_args = []

        def on_retry(exception, attempt, delay):
            callback_args.append((attempt, delay))

        retry = RetryPolicy(
            max_retries=2, initial_delay=0.01,
            on_retry=on_retry,
        )
        count = [0]

        def flaky():
            count[0] += 1
            if count[0] < 3:
                raise ValueError("nope")
            return "ok"

        retry.call(flaky)
        assert len(callback_args) == 2


# ============================================================================
#  SlaTracker 测试
# ============================================================================

class TestSlaTracker:
    """SlaTracker SLA 追踪器测试"""

    def test_record_up_and_get_sla(self):
        """记录正常并获取 SLA"""
        tracker = SlaTracker(window_days=30)
        for _ in range(99):
            tracker.record_up("data_server")
        tracker.record_down("data_server", reason="crash")

        sla = tracker.get_sla("data_server")
        assert sla == 0.99

    def test_record_degraded(self):
        """记录降级"""
        tracker = SlaTracker(window_days=30)
        tracker.record_up("svc")
        tracker.record_degraded("svc")

        sla = tracker.get_sla("svc")
        # 1 up + 0.5 degraded / 2 total = 0.75
        assert sla == 0.75

    def test_is_sla_breached(self):
        """SLA 是否跌破"""
        tracker = SlaTracker(target_sla=0.999, window_days=30)
        for _ in range(10):
            tracker.record_up("svc")
        # 10 up / 10 = 100% → 未跌破
        assert not tracker.is_sla_breached("svc")

        tracker.record_down("svc")
        # 10 up / 11 ≈ 90.9% → 跌破 99.9%
        assert tracker.is_sla_breached("svc")

    def test_is_99_9_available(self):
        """99.9% 可用性检查"""
        tracker = SlaTracker(window_days=30)
        # 1000 up, 0 down = 100%
        for _ in range(1000):
            tracker.record_up("svc")
        assert tracker.is_99_9_available("svc")

        # 999 up, 1 down = 99.9% → 达标
        tracker.record_down("svc")
        assert tracker.is_99_9_available("svc")

    def test_get_sla_report(self):
        """SLA 报告"""
        tracker = SlaTracker(window_days=30)
        tracker.record_up("data_server", response_time_ms=12.5)
        tracker.record_up("data_server", response_time_ms=8.3)
        tracker.record_down("data_server", reason="timeout")
        tracker.record_up("api_gateway")

        report = tracker.get_sla_report()
        assert report["target_sla"] == 0.999
        assert "data_server" in report["services"]
        assert "api_gateway" in report["services"]
        assert "all_meet_target" in report
        ds = report["services"]["data_server"]
        assert ds["up"] == 2
        assert ds["down"] == 1

    def test_get_sla_new_service(self):
        """新服务默认 100%"""
        tracker = SlaTracker()
        assert tracker.get_sla("new_service") == 1.0

    def test_reset(self):
        """重置"""
        tracker = SlaTracker()
        tracker.record_up("svc")
        tracker.record_down("svc")

        tracker.reset("svc")
        assert tracker.get_sla("svc") == 1.0

    def test_reset_all(self):
        """重置全部"""
        tracker = SlaTracker()
        tracker.record_up("svc")
        tracker.reset()
        assert tracker.get_sla("svc") == 1.0

    def test_response_time_tracking(self):
        """响应时间追踪"""
        tracker = SlaTracker()
        tracker.record_up("svc", response_time_ms=10.0)
        tracker.record_up("svc", response_time_ms=20.0)

        report = tracker.get_sla_report()
        assert abs(report["services"]["svc"]["avg_response_ms"] - 15.0) < 0.01


# ============================================================================
#  AutoHealingManager 测试
# ============================================================================

class TestAutoHealingManager:
    """AutoHealingManager 自动恢复管理器测试"""

    def test_no_actions_needed_when_healthy(self):
        """健康时无需自愈"""
        manager = AutoHealingManager(check_health=lambda: True)
        results = manager.heal()
        assert len(results) == 0

    def test_healing_action_success(self):
        """自愈动作成功"""
        healed = [False]

        def make_healthy():
            healed[0] = True

        manager = AutoHealingManager(check_health=lambda: healed[0])
        manager.register_action(HealingAction(
            name="fix_it",
            action=make_healthy,
            priority=1,
        ))

        results = manager.heal()
        assert len(results) == 1
        assert results[0].success is True
        assert results[0].action_name == "fix_it"

    def test_healing_action_failure(self):
        """自愈动作失败"""
        self_heal_count = [0]

        def check():
            self_heal_count[0] += 1
            return self_heal_count[0] >= 3

        def always_fail():
            raise RuntimeError("cannot fix")

        manager = AutoHealingManager(check_health=check)
        manager.register_action(HealingAction(
            name="broken_fix",
            action=always_fail,
            priority=1,
        ))

        results = manager.heal(max_rounds=3)
        assert any(not r.success for r in results)

    def test_multiple_actions_priority_order(self):
        """按优先级执行"""
        execution_order = []

        manager = AutoHealingManager(check_health=lambda: len(execution_order) >= 2)

        manager.register_action(HealingAction(
            name="low", action=lambda: execution_order.append("low"), priority=10,
        ))
        manager.register_action(HealingAction(
            name="high", action=lambda: execution_order.append("high"), priority=1,
        ))

        manager.heal()
        assert execution_order[0] == "high"
        assert execution_order[1] == "low"

    def test_register_and_unregister(self):
        """注册与移除"""
        manager = AutoHealingManager()

        manager.register_action(HealingAction(
            name="temp", action=lambda: None,
        ))
        assert "temp" in manager._actions

        manager.unregister_action("temp")
        assert "temp" not in manager._actions

    def test_get_history(self):
        """获取历史"""
        manager = AutoHealingManager(check_health=lambda: False)
        manager.register_action(HealingAction(
            name="fix", action=lambda: None, priority=1,
        ))

        manager.heal(max_rounds=1)
        history = manager.get_history()
        assert len(history) >= 1

    def test_get_stats(self):
        """统计信息"""
        manager = AutoHealingManager()
        stats = manager.get_stats()
        assert "total_healings" in stats
        assert "success_rate" in stats

    def test_is_healthy_default(self):
        """默认健康检查"""
        manager = AutoHealingManager()
        assert manager.is_healthy is True

    def test_check_health_exception(self):
        """健康检查异常"""
        def broken_check():
            raise RuntimeError("check failed")

        manager = AutoHealingManager(check_health=broken_check)
        assert manager.is_healthy is False


# ============================================================================
#  with_resilience 测试
# ============================================================================

class TestWithResilience:
    """with_resilience 组合装饰器测试"""

    def test_basic_combination(self):
        """基本组合"""
        breaker = CircuitBreaker(name="wr_test", failure_threshold=5)
        retry = RetryPolicy(max_retries=2, initial_delay=0.01)

        @with_resilience(breaker=breaker, retry=retry)
        def simple():
            return 42

        result = simple()
        assert result == 42

    def test_only_breaker(self):
        """仅断路器"""
        breaker = CircuitBreaker(name="bo_test", failure_threshold=5)

        @with_resilience(breaker=breaker)
        def add(a, b):
            return a + b

        assert add(1, 2) == 3

    def test_only_retry(self):
        """仅重试"""
        retry = RetryPolicy(max_retries=2, initial_delay=0.01)
        count = [0]

        @with_resilience(retry=retry)
        def flaky():
            count[0] += 1
            if count[0] < 2:
                raise ValueError("nope")
            return "ok"

        assert flaky() == "ok"
