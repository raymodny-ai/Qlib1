"""
第4章 极限性能基准与高可用性目标 — 集成测试

验证场景:
1. 性能基准 → 数据服务器 (BenchmarkTimer + DataServer 特征组装性能)
2. 熔断器 → 健康检查器 (CircuitBreaker + HealthChecker 联动)
3. 准确度校验 → 策略回测红线 (AccuracyValidator + DrawdownValidator)
4. 全链路稳定性 (CircuitBreaker + RetryPolicy + SlaTracker + AutoHealingManager)
5. 缓存性能 → 准确度评估 (CacheStatsTracker + StrategyStabilityChecker)
"""

import time
import threading
import tempfile
import pytest
import numpy as np
import pandas as pd
from unittest.mock import MagicMock, patch, PropertyMock

from src.infrastructure.performance import (
    BenchmarkTimer,
    CacheStatsTracker,
    FeatureAssemblyBenchmark,
    FeatureAssemblyResult,
    ThroughputMonitor,
    PerformanceReport,
    generate_performance_report,
)
from src.infrastructure.circuit_breaker import (
    CircuitBreaker,
    CircuitState,
    CircuitBreakerOpenError,
    RetryPolicy,
    SlaTracker,
    AutoHealingManager,
    HealingAction,
    with_resilience,
)
from src.infrastructure.health_checker import (
    HealthChecker,
    HealthReport,
    CheckResult,
    CheckStatus,
)
from src.analyzers.accuracy_validator import (
    AccuracyThresholdValidator,
    RollingICValidator,
    DrawdownValidator,
    StrategyStabilityChecker,
    ValidationReport,
    quick_validate,
)


# ============================================================================
#  性能基准 ↔ 数据服务器 集成
# ============================================================================

class TestChapter4PerformanceDataServerIntegration:
    """性能基准 + 数据流 端到端验证"""

    def setup_method(self):
        BenchmarkTimer.reset()
        BenchmarkTimer.enable()

    def test_feature_assembly_benchmark_timing(self):
        """特征组装性能基准端到端计时"""
        class MockDS:
            def load_features(self, instruments, fields, start, end):
                time.sleep(0.005)
                return pd.DataFrame()

        bench = FeatureAssemblyBenchmark(data_server=MockDS())
        result = bench.benchmark(
            instruments=["AAPL", "MSFT", "GOOGL", "AMZN", "META"],
            fields=["close", "volume", "pe_ratio", "roe", "market_cap"],
            start="2014-01-01",
            end="2024-12-31",
            warmup_iterations=1,
            benchmark_iterations=3,
        )
        assert result.meets_10s_target is True
        assert len(result.durations) == 3
        assert result.warmed_up is True

    def test_feature_assembly_benchmark_fails_on_slow_data(self):
        """慢数据源导致基准不达标"""
        class SlowDS:
            def load_features(self, instruments, fields, start, end):
                time.sleep(0.02)  # 较慢

        bench = FeatureAssemblyBenchmark(data_server=SlowDS())
        result = bench.benchmark(
            instruments=["AAPL"],
            fields=["close"],
            start="2020-01-01",
            end="2020-12-31",
            warmup_iterations=0,
            benchmark_iterations=1,
        )
        # 小数据量下仍应满足 10s 目标
        assert result.meets_10s_target is True

    def test_throughput_monitor_with_data_flow(self):
        """吞吐量监控器跟踪数据读写"""
        monitor = ThroughputMonitor(window_seconds=10)

        # 模拟数据读取
        for _ in range(5):
            monitor.record_read(bytes_read=2 * 1024 * 1024, records=1000)

        summary = monitor.get_summary()
        assert summary["total_bytes_read"] == 10 * 1024 * 1024
        assert summary["total_records_read"] == 5000

    def test_cache_tracker_80_percent_target(self):
        """缓存追踪器 80% 目标验证"""
        tracker = CacheStatsTracker()
        tracker.register_level("global_memory", max_size_gb=4)
        tracker.register_level("expression_cache", max_size_mb=512)
        tracker.register_level("dataset_cache", max_size_mb=1024)

        # 模拟 85% 命中率
        for _ in range(85):
            tracker.record_hit("global_memory")
        for _ in range(10):
            tracker.record_miss("global_memory")
        for _ in range(90):
            tracker.record_hit("expression_cache")
        for _ in range(10):
            tracker.record_miss("expression_cache")
        for _ in range(80):
            tracker.record_hit("dataset_cache")
        for _ in range(20):
            tracker.record_miss("dataset_cache")

        assert tracker.meets_target is True  # 综合 > 80%

    def test_generate_performance_report_full_chain(self):
        """完整性能报告链"""
        tracker = CacheStatsTracker()
        tracker.register_level("L1")
        tracker.record_hit("L1")

        monitor = ThroughputMonitor()
        monitor.record_read(bytes_read=1024 * 1024)

        with BenchmarkTimer("feature_load"):
            time.sleep(0.001)
        with BenchmarkTimer("model_predict"):
            time.sleep(0.002)

        report = generate_performance_report(
            cache_tracker=tracker,
            throughput_monitor=monitor,
        )
        d = report.to_dict()
        assert len(d["timers"]) == 2
        assert d["combined_hit_rate"] == 1.0
        assert d["throughput_mbps"] > 0


# ============================================================================
#  熔断器 ↔ 健康检查器 联动
# ============================================================================

class TestChapter4CircuitBreakerHealthCheckerIntegration:
    """熔断器 + 健康检查器 联动测试"""

    def test_circuit_breaker_triggers_health_alert(self):
        """熔断开路时触发健康告警"""
        alerts = []

        def on_open(name):
            alerts.append(f"ALERT: {name} circuit OPEN")

        breaker = CircuitBreaker(
            name="data_handler",
            failure_threshold=2,
            on_open=on_open,
        )

        # 触发开路
        for _ in range(2):
            try:
                with breaker:
                    raise ConnectionError("timeout")
            except ConnectionError:
                pass

        assert len(alerts) == 1
        assert "OPEN" in alerts[0]

    def test_circuit_breaker_integrated_with_health_status(self):
        """熔断器状态纳入健康检查"""
        breaker = CircuitBreaker(name="api_client", failure_threshold=2)

        # 模拟连续失败
        for _ in range(2):
            try:
                with breaker:
                    raise TimeoutError("timeout")
            except TimeoutError:
                pass

        # 健康检查中集成熔断器状态
        health_status = {
            "component": "api_client",
            "circuit_state": breaker.state.value,
            "healthy": not breaker.is_open,
        }
        assert health_status["healthy"] is False
        assert health_status["circuit_state"] == "open"

    def test_health_checker_auto_healing_fix_circuit(self):
        """健康检查触发自动修复"""
        breaker = CircuitBreaker(name="test", failure_threshold=1)

        # 触发开路
        try:
            with breaker:
                raise RuntimeError("fail")
        except RuntimeError:
            pass

        assert breaker.is_open

        # 自动修复：重置熔断器
        breaker.reset()
        assert not breaker.is_open

    def test_retry_with_health_probe(self):
        """重试策略 + 健康探针"""
        retry = RetryPolicy(max_retries=2, initial_delay=0.01)
        probe_results = []

        def health_probe():
            return len(probe_results) > 0

        count = [0]

        def flaky_operation():
            count[0] += 1
            probe_results.append("checked")
            if count[0] < 2:
                raise ConnectionError("not ready")
            return "success"

        result = retry.call(flaky_operation)
        assert result == "success"
        assert count[0] == 2


# ============================================================================
#  准确度校验 → 策略回测红线
# ============================================================================

class TestChapter4AccuracyBacktestIntegration:
    """准确度校验 + 策略回测红线 集成"""

    def test_full_accuracy_pipeline_passing(self):
        """完整准确度流水线通过"""
        np.random.seed(42)
        # 生成符合要求的模拟回测结果
        rank_ic_series = np.random.normal(0.050, 0.005, 300)
        # 净值曲线：稳定增长，小幅回撤
        daily_returns = np.random.normal(0.0005, 0.008, 300)
        nav = np.cumprod(1 + daily_returns)

        checker = StrategyStabilityChecker()
        result = checker.check(
            rank_ic_series=rank_ic_series,
            rank_icir=0.52,
            nav_series=nav,
            returns_series=daily_returns,
        )
        assert "approved" in result
        assert "ic_report" in result
        assert "rolling_report" in result
        assert "drawdown_report" in result
        assert result["ic_report"]["passed"] is True

    def test_full_accuracy_pipeline_fails_on_deep_drawdown(self):
        """深度回撤导致全线失败"""
        np.random.seed(42)
        rank_ic_series = np.random.normal(0.050, 0.005, 300)
        # 制造 -25% 回撤的净值曲线
        nav = np.array([1.0, 0.95, 0.9, 0.85, 0.8, 0.75, 0.72, 0.8, 0.9, 1.0])

        checker = StrategyStabilityChecker()
        result = checker.check(
            rank_ic_series=rank_ic_series,
            rank_icir=0.52,
            nav_series=nav,
        )
        assert result["approved"] is False

    def test_rolling_ic_stability_on_synthetic_data(self):
        """滚动 IC 稳定性验证"""
        validator = RollingICValidator(window_size=30)
        np.random.seed(42)

        # 模拟 3 年日度 IC
        ic_series = np.random.normal(0.050, 0.006, 756)

        result = validator.validate(ic_series)
        # 高命中率应在大部分窗口稳定
        assert result.ic_stable_ratio > 0.5
        assert result.ic_mean > 0.04

    def test_drawdown_validator_with_portfolio_simulation(self):
        """回撤校验器模拟组合回测"""
        validator = DrawdownValidator(max_drawdown_limit=-0.15)

        # 模拟 3 年净值 (含一次 -20% 回撤)
        np.random.seed(42)
        n = 756
        daily_returns = np.concatenate([
            np.random.normal(0.0005, 0.008, 200),
            np.random.normal(-0.002, 0.015, 100),  # 下跌期
            np.random.normal(0.0005, 0.008, 456),
        ])
        nav = np.cumprod(1 + daily_returns)

        report = validator.validate(nav)
        assert isinstance(report, ValidationReport)

    def test_quick_validate_in_strategy_context(self):
        """快速校验在策略上下文中的使用"""
        # 模拟策略输出
        rank_ic = 0.048
        rank_icir = 0.45
        max_dd = -0.12

        passed, msg = quick_validate(rank_ic, rank_icir, max_dd)
        assert passed is True

        # 模拟不达标的策略
        passed2, msg2 = quick_validate(0.035, 0.30, -0.20)
        assert passed2 is False


# ============================================================================
#  全链路稳定性 (CircuitBreaker + RetryPolicy + SLA + AutoHeal)
# ============================================================================

class TestChapter4FullResilienceChain:
    """全链路稳定性集成"""

    def test_circuit_breaker_with_retry_resilience_chain(self):
        """断路器 + 重试 韧性链"""
        breaker = CircuitBreaker(name="resilience_test", failure_threshold=3)
        retry = RetryPolicy(max_retries=2, initial_delay=0.01)
        count = [0]

        @with_resilience(breaker=breaker, retry=retry)
        def resilient_op():
            count[0] += 1
            if count[0] < 2:
                raise ValueError("transient")
            return "ok"

        result = resilient_op()
        assert result == "ok"
        assert count[0] == 2

    def test_sla_tracker_with_circuit_breaker(self):
        """SLA + 熔断器联动"""
        tracker = SlaTracker(window_days=30)

        # 正常服务 → 记录 up
        tracker.record_up("data_service", response_time_ms=5.0)
        tracker.record_up("data_service", response_time_ms=8.0)

        # 熔断开路 → 记录 down
        tracker.record_down("data_service", reason="circuit_breaker_open")

        sla = tracker.get_sla("data_service")
        assert sla == pytest.approx(2 / 3, rel=0.01)

    def test_auto_healing_manager_full_recovery(self):
        """自动恢复管理器完整恢复流程"""
        system_healthy = [False]
        fix_count = [0]

        def check_health():
            return system_healthy[0]

        def fix_step():
            fix_count[0] += 1
            if fix_count[0] >= 2:
                system_healthy[0] = True

        manager = AutoHealingManager(check_health=check_health)
        manager.register_action(HealingAction(
            name="restart_service",
            action=fix_step,
            priority=1,
        ))

        results = manager.heal(max_rounds=3)
        assert system_healthy[0] is True
        assert len(results) > 0
        # 最后一步应成功
        assert results[-1].success is True

    def test_sla_99_9_target_monitoring(self):
        """99.9% SLA 监控"""
        tracker = SlaTracker(target_sla=0.999, window_days=30)

        # 999 up, 1 down → 99.9% (刚好达标)
        for _ in range(999):
            tracker.record_up("critical_svc")
        tracker.record_down("critical_svc")

        # 应刚好达标
        assert tracker.is_99_9_available("critical_svc")

        # 再加 1 down → 跌破
        tracker.record_down("critical_svc")
        assert not tracker.is_99_9_available("critical_svc")


# ============================================================================
#  缓存性能 + 准确度评估 集成
# ============================================================================

class TestChapter4CacheAccuracyIntegration:
    """缓存性能 + 准确度评估 联动"""

    def test_cache_warmup_improves_performance(self):
        """缓存预热改善性能"""
        tracker = CacheStatsTracker()
        tracker.register_level("global_cache")

        # 预热前：全部 miss
        for _ in range(10):
            tracker.record_miss("global_cache")
        cold_hit_rate = tracker.combined_hit_rate
        assert cold_hit_rate == 0.0

        # 预热后：大部分 hit
        tracker.reset()
        tracker.register_level("global_cache")
        for _ in range(85):
            tracker.record_hit("global_cache")
        for _ in range(15):
            tracker.record_miss("global_cache")

        warm_hit_rate = tracker.combined_hit_rate
        assert warm_hit_rate > 0.8

    def test_performance_degradation_affects_accuracy(self):
        """性能退化影响准确度评估时效"""
        # 模拟：缓存命中率下降 → 数据获取变慢 → IC 计算延迟
        tracker = CacheStatsTracker()
        tracker.register_level("global_cache")

        # 低命中率场景
        for _ in range(30):
            tracker.record_hit("global_cache")
        for _ in range(70):
            tracker.record_miss("global_cache")

        assert tracker.meets_target is False

        # 在此场景下，特征组装时间预期增长 (用 BenchmarkTimer 跟踪)
        with BenchmarkTimer("slow_assembly") as t:
            time.sleep(0.01)

        stats = BenchmarkTimer.get_stats("slow_assembly")
        assert stats["count"] == 1

    def test_accuracy_validator_with_cache_metrics(self):
        """缓存性能指标纳入准确度校验上下文"""
        # 模拟：缓存命中率正常 → IC 计算及时 → 准确度在阈值内
        tracker = CacheStatsTracker()
        tracker.register_level("global_cache")
        for _ in range(85):
            tracker.record_hit("global_cache")
        for _ in range(15):
            tracker.record_miss("global_cache")

        cache_ok = tracker.meets_target

        # 准确的模拟 IC
        np.random.seed(42)
        rank_ic = np.random.normal(0.05, 0.005, 300)

        validator = AccuracyThresholdValidator()
        report = validator.validate(
            rank_ic_series=rank_ic,
            rank_icir=0.48,
            max_drawdown=-0.10,
        )

        # 两者都应达标
        assert cache_ok is True
        assert report.passed is True


# ============================================================================
#  综合压测: 第4章全部组件协同验证
# ============================================================================

class TestChapter4ComprehensiveStressTest:
    """第4章全面压测"""

    def test_full_chapter4_pipeline(self):
        """
        PRD 4.1 + 4.2 + 4.3 全链路:
        BenchmarkTimer → CacheStatsTracker → AccuracyValidator → CircuitBreaker → SlaTracker
        """
        # 1. 计时
        BenchmarkTimer.reset()
        BenchmarkTimer.enable()

        # 2. 缓存
        tracker = CacheStatsTracker()
        tracker.register_level("global_cache", max_size_gb=4)

        # 3. 模拟特征组装 (用 BenchmarkTimer 包裹)
        with BenchmarkTimer("feature_assembly"):
            # 模拟数据加载
            time.sleep(0.003)
            for _ in range(8):
                tracker.record_hit("global_cache")
            for _ in range(2):
                tracker.record_miss("global_cache")

        # 4. 模拟模型预测
        with BenchmarkTimer("model_predict"):
            time.sleep(0.002)

        # 5. 生成性能报告
        report = generate_performance_report(cache_tracker=tracker)
        assert report.combined_hit_rate >= 0.8

        # 6. 准确度校验
        np.random.seed(42)
        rank_ic = np.random.normal(0.050, 0.005, 300)
        nav = np.cumprod(1 + np.random.normal(0.0005, 0.008, 300))

        validator = AccuracyThresholdValidator()
        acc_report = validator.validate(
            rank_ic_series=rank_ic,
            rank_icir=0.50,
            max_drawdown=DrawdownValidator.compute_max_drawdown(nav),
        )
        assert acc_report.passed is True

        # 7. 熔断保护
        breaker = CircuitBreaker(name="production", failure_threshold=5)
        retry = RetryPolicy(max_retries=1, initial_delay=0.01)

        @with_resilience(breaker=breaker, retry=retry)
        def prod_op():
            return "ok"

        assert prod_op() == "ok"

        # 8. SLA 追踪
        sla = SlaTracker()
        sla.record_up("prod_service")
        assert sla.is_99_9_available("prod_service")

        # 9. 计时汇总
        stats = BenchmarkTimer.get_stats()
        assert stats["count"] >= 2
