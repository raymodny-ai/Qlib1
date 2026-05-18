"""
第4章 性能基准模块 单元测试

覆盖:
- BenchmarkTimer: 上下文管理器 / 装饰器 / 统计
- CacheStatsTracker: 注册 / 命中 / 未命中 / 驱逐 / 综合命中率
- CacheLevelStats: 数据类属性
- FeatureAssemblyBenchmark: 基准测试 / 数据量估算
- ThroughputMonitor: 读写吞吐量 / 快照管理
- PerformanceReport: 综合报告生成
"""

import time
import threading
import pytest
import numpy as np

from src.infrastructure.performance import (
    BenchmarkTimer,
    TimerRecord,
    CacheStatsTracker,
    CacheLevelStats,
    FeatureAssemblyBenchmark,
    FeatureAssemblyResult,
    ThroughputMonitor,
    ThroughputSnapshot,
    PerformanceReport,
    generate_performance_report,
)


# ============================================================================
#  BenchmarkTimer 测试
# ============================================================================

class TestBenchmarkTimer:
    """BenchmarkTimer 计时器测试"""

    def setup_method(self):
        BenchmarkTimer.reset()
        BenchmarkTimer.enable()

    def test_context_manager_basic(self):
        """基本上下文管理器计时"""
        with BenchmarkTimer("test_op") as t:
            time.sleep(0.01)
        assert t.duration_ms > 0
        assert t.name == "test_op"

    def test_context_manager_nested(self):
        """嵌套计时"""
        with BenchmarkTimer("outer") as outer:
            time.sleep(0.005)
            with BenchmarkTimer("inner") as inner:
                time.sleep(0.005)
        assert outer.duration_ms > 0
        assert inner.duration_ms > 0
        assert outer.name == "outer"
        assert inner.name == "inner"

    def test_metadata_attachment(self):
        """元数据附着"""
        with BenchmarkTimer("with_meta", metadata={"size": 1024}) as t:
            time.sleep(0.001)
        assert t.metadata["size"] == 1024
        assert "function" not in t.metadata

    def test_decorator_mode(self):
        """装饰器模式"""

        @BenchmarkTimer.decorate("decorated_func")
        def slow_add(a, b):
            time.sleep(0.005)
            return a + b

        result = slow_add(1, 2)
        assert result == 3

        stats = BenchmarkTimer.get_stats("decorated_func")
        assert stats["count"] == 1

    def test_disable_enable(self):
        """启用/禁用开关"""
        BenchmarkTimer.disable()
        with BenchmarkTimer("disabled_op"):
            time.sleep(0.001)
        stats = BenchmarkTimer.get_stats("disabled_op")
        assert stats["count"] == 0

        BenchmarkTimer.enable()
        with BenchmarkTimer("enabled_op"):
            time.sleep(0.001)
        stats = BenchmarkTimer.get_stats("enabled_op")
        assert stats["count"] == 1

    def test_statistics_aggregation(self):
        """统计聚合"""
        for _ in range(5):
            with BenchmarkTimer("agg_test"):
                time.sleep(0.002)

        stats = BenchmarkTimer.get_stats("agg_test")
        assert stats["count"] == 5
        assert stats["mean_ms"] > 0
        assert stats["min_ms"] > 0
        assert stats["max_ms"] > 0
        assert stats["p50_ms"] > 0
        assert stats["p95_ms"] > 0
        assert stats["p99_ms"] > 0

    def test_statistics_empty(self):
        """空统计"""
        BenchmarkTimer.reset()
        stats = BenchmarkTimer.get_stats("nonexistent")
        assert stats["count"] == 0
        assert stats["mean_ms"] == 0.0

    def test_get_all_stats(self):
        """获取全部统计"""
        with BenchmarkTimer("op_a"):
            time.sleep(0.001)
        with BenchmarkTimer("op_b"):
            time.sleep(0.002)

        stats = BenchmarkTimer.get_stats()
        assert stats["count"] == 2

    def test_get_all_records(self):
        """获取全部记录"""
        BenchmarkTimer.reset()
        with BenchmarkTimer("r1"):
            time.sleep(0.001)
        records = BenchmarkTimer.get_all_records()
        assert "r1" in records
        assert len(records["r1"]) == 1

    def test_set_warmup_complete(self):
        """预热完成标记"""
        assert BenchmarkTimer._warmup_complete is False
        BenchmarkTimer.set_warmup_complete()
        assert BenchmarkTimer._warmup_complete is True
        BenchmarkTimer._warmup_complete = False  # 恢复

    def test_timer_record_duration_s(self):
        """TimerRecord duration_s 属性"""
        record = TimerRecord(name="test", duration_ms=1500.0)
        assert record.duration_s == 1.5

    def test_exception_propagation(self):
        """异常正常传播"""
        BenchmarkTimer.reset()
        try:
            with BenchmarkTimer("will_fail"):
                raise ValueError("test_error")
        except ValueError:
            pass
        # 异常不应被吞没，记录应仍存在
        stats = BenchmarkTimer.get_stats("will_fail")
        assert stats["count"] == 1

    def test_thread_safety_records(self):
        """线程安全记录"""
        BenchmarkTimer.reset()

        def timed_op():
            with BenchmarkTimer("thread_op"):
                time.sleep(0.001)

        threads = [threading.Thread(target=timed_op) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        stats = BenchmarkTimer.get_stats("thread_op")
        assert stats["count"] == 10


# ============================================================================
#  CacheStatsTracker 测试
# ============================================================================

class TestCacheStatsTracker:
    """CacheStatsTracker 缓存统计测试"""

    def test_register_level(self):
        """注册缓存层级"""
        tracker = CacheStatsTracker()
        tracker.register_level("global_cache", max_size_gb=4)
        stats = tracker.get_level_stats("global_cache")
        assert stats is not None
        assert stats.max_size_bytes == 4 * 1024 * 1024 * 1024
        assert stats.level_name == "global_cache"

    def test_record_hit_and_miss(self):
        """记录命中和未命中"""
        tracker = CacheStatsTracker()
        tracker.register_level("L1", max_size_mb=512)

        tracker.record_hit("L1")
        tracker.record_hit("L1")
        tracker.record_miss("L1")

        stats = tracker.get_level_stats("L1")
        assert stats.hits == 2
        assert stats.misses == 1
        assert stats.total_accesses == 3

    def test_record_eviction(self):
        """记录驱逐"""
        tracker = CacheStatsTracker()
        tracker.register_level("L1", max_size_mb=1024)
        tracker.record_hit("L1", size_bytes=500 * 1024 * 1024)
        tracker.record_eviction("L1", freed_bytes=200 * 1024 * 1024)

        stats = tracker.get_level_stats("L1")
        assert stats.evictions == 1
        assert stats.size_bytes == 300 * 1024 * 1024

    def test_update_size(self):
        """更新缓存大小"""
        tracker = CacheStatsTracker()
        tracker.register_level("L1", max_size_mb=1024)
        tracker.update_size("L1", 512 * 1024 * 1024)
        stats = tracker.get_level_stats("L1")
        assert stats.size_bytes == 512 * 1024 * 1024

    def test_fill_ratio(self):
        """填充率计算"""
        tracker = CacheStatsTracker()
        tracker.register_level("L1", max_size_mb=1024)
        tracker.update_size("L1", 256 * 1024 * 1024)
        stats = tracker.get_level_stats("L1")
        assert stats.fill_ratio == 0.25

    def test_combined_hit_rate(self):
        """综合命中率"""
        tracker = CacheStatsTracker()
        tracker.register_level("L1", max_size_mb=512)
        tracker.register_level("L2", max_size_gb=4)

        tracker.record_hit("L1")
        tracker.record_hit("L1")
        tracker.record_miss("L1")
        tracker.record_hit("L2")
        tracker.record_miss("L2")

        # hits=3, accesses=5
        assert tracker.combined_hit_rate == pytest.approx(0.6)

    def test_combined_hit_rate_empty(self):
        """空追踪器"""
        tracker = CacheStatsTracker()
        assert tracker.combined_hit_rate == 0.0

    def test_meets_target(self):
        """80% 目标检查"""
        tracker = CacheStatsTracker()
        tracker.register_level("L1")
        # 80% 命中率
        for _ in range(80):
            tracker.record_hit("L1")
        for _ in range(20):
            tracker.record_miss("L1")
        assert tracker.meets_target is True

        # 加一次 miss，变为 80/101 ≈ 79.2%
        tracker.record_miss("L1")
        assert tracker.meets_target is False

    def test_get_summary(self):
        """摘要输出"""
        tracker = CacheStatsTracker()
        tracker.register_level("L1")
        tracker.record_hit("L1")
        tracker.record_miss("L1")

        summary = tracker.get_summary()
        assert "combined_hit_rate" in summary
        assert "meets_target" in summary
        assert "L1" in summary["levels"]
        assert summary["levels"]["L1"]["hit_rate"] == 0.5

    def test_reset(self):
        """重置统计"""
        tracker = CacheStatsTracker()
        tracker.register_level("L1")
        tracker.record_hit("L1")
        tracker.record_hit("L1")
        tracker.record_miss("L1")
        tracker.reset()

        stats = tracker.get_level_stats("L1")
        assert stats.hits == 0
        assert stats.misses == 0
        assert stats.size_bytes == 0

    def test_get_nonexistent_level(self):
        """获取不存在的层级"""
        tracker = CacheStatsTracker()
        assert tracker.get_level_stats("ghost") is None

    def test_cache_level_stats_data_class(self):
        """CacheLevelStats 数据类"""
        cls = CacheLevelStats("test")
        assert cls.hit_rate == 0.0
        assert cls.miss_rate == 0.0
        assert cls.fill_ratio == 0.0

    def test_size_not_exceed_max(self):
        """size 不超过 max"""
        tracker = CacheStatsTracker()
        tracker.register_level("L1", max_size_mb=1)
        tracker.record_hit("L1", size_bytes=10 * 1024 * 1024)  # 10MB > 1MB max
        stats = tracker.get_level_stats("L1")
        assert stats.size_bytes <= stats.max_size_bytes


# ============================================================================
#  FeatureAssemblyBenchmark 测试
# ============================================================================

class MockDataServer:
    """Mock DataServer for benchmarking"""
    def load_features(self, instruments, fields, start, end):
        time.sleep(0.005)
        return None


class TestFeatureAssemblyBenchmark:
    """FeatureAssemblyBenchmark 特征组装基准测试"""

    def test_benchmark_basic(self):
        """基本基准测试"""
        bench = FeatureAssemblyBenchmark(data_server=MockDataServer())
        result = bench.benchmark(
            instruments=["AAPL", "MSFT", "GOOGL"],
            fields=["close", "volume"],
            start="2018-01-01",
            end="2023-12-31",
            warmup_iterations=1,
            benchmark_iterations=2,
        )
        assert isinstance(result, FeatureAssemblyResult)
        assert result.instrument_count == 3
        assert result.field_count == 2
        assert result.warmed_up is True
        assert len(result.durations) == 2
        assert result.mean_duration_s > 0

    def test_meets_10s_target(self):
        """10s 目标检查"""
        result = FeatureAssemblyResult(
            instrument_count=100,
            field_count=20,
            date_range_start="2010-01-01",
            date_range_end="2023-12-31",
            durations=[0.5, 0.6, 0.7],
            warmed_up=True,
        )
        assert result.meets_10s_target is True

    def test_fails_10s_target(self):
        """超过 10s 目标"""
        result = FeatureAssemblyResult(
            instrument_count=100,
            field_count=20,
            date_range_start="2010-01-01",
            date_range_end="2023-12-31",
            durations=[11.0, 12.0],
            warmed_up=True,
        )
        assert result.meets_10s_target is False

    def test_to_dict(self):
        """to_dict 输出"""
        result = FeatureAssemblyResult(
            instrument_count=10,
            field_count=5,
            date_range_start="2020-01-01",
            date_range_end="2020-12-31",
            durations=[0.5, 0.6],
            cache_hit_rates=[0.85, 0.88],
            warmed_up=True,
        )
        d = result.to_dict()
        assert d["instrument_count"] == 10
        assert d["field_count"] == 5
        assert d["warmed_up"] is True
        assert d["meets_10s_target"] is True
        assert abs(d["mean_hit_rate"] - 0.865) < 0.001

    def test_estimate_data_volume(self):
        """数据量估算"""
        # 100 stocks * 20 fields * 2520 days (10yr) * 4 bytes
        mb = FeatureAssemblyBenchmark.estimate_data_volume(
            instruments=100, fields=20, days=2520,
        )
        expected = 100 * 20 * 2520 * 4 / (1024 * 1024)
        assert mb == pytest.approx(expected)

    def test_benchmark_without_dataserer(self):
        """无 DataServer 时的优雅降级"""
        bench = FeatureAssemblyBenchmark(data_server=None)
        result = bench.benchmark(
            instruments=["AAPL"],
            fields=["close"],
            start="2020-01-01",
            end="2020-12-31",
            warmup_iterations=0,
            benchmark_iterations=1,
        )
        assert len(result.durations) == 1

    def test_benchmark_with_cache_tracker(self):
        """带 CacheStatsTracker 的基准测试"""
        tracker = CacheStatsTracker()
        tracker.register_level("global_cache")
        bench = FeatureAssemblyBenchmark(
            data_server=MockDataServer(),
            cache_tracker=tracker,
        )
        result = bench.benchmark(
            instruments=["AAPL"],
            fields=["close"],
            start="2020-01-01",
            end="2020-12-31",
            warmup_iterations=1,
            benchmark_iterations=2,
        )
        assert len(result.cache_hit_rates) == 2

    def test_empty_durations(self):
        """空 durations"""
        result = FeatureAssemblyResult(
            instrument_count=0,
            field_count=0,
            date_range_start="",
            date_range_end="",
        )
        assert result.mean_duration_s == 0.0
        assert result.p95_duration_s == 0.0
        assert result.mean_hit_rate == 0.0

    def test_p95_duration(self):
        """P95 延迟计算"""
        result = FeatureAssemblyResult(
            instrument_count=10, field_count=5,
            date_range_start="", date_range_end="",
            durations=list(range(1, 21)),  # 1..20
        )
        # P95 of 1-20 = 19.05 (接近 20)
        assert 18.0 <= result.p95_duration_s <= 20.0


# ============================================================================
#  ThroughputMonitor 测试
# ============================================================================

class TestThroughputMonitor:
    """ThroughputMonitor 吞吐量监控测试"""

    def test_record_read(self):
        """记录读操作"""
        monitor = ThroughputMonitor(window_seconds=60)
        monitor.record_read(bytes_read=1024 * 1024 * 10, records=100)
        assert monitor._total_bytes_read == 1024 * 1024 * 10

    def test_record_write(self):
        """记录写操作"""
        monitor = ThroughputMonitor(window_seconds=60)
        monitor.record_write(bytes_written=1024 * 1024 * 5, records=50)
        assert monitor._total_bytes_written == 1024 * 1024 * 5

    def test_read_throughput_mbps(self):
        """读吞吐量计算"""
        monitor = ThroughputMonitor(window_seconds=10)
        # 10 MB in 10 seconds = 1 MB/s
        monitor.record_read(bytes_read=10 * 1024 * 1024)
        assert monitor.read_throughput_mbps == pytest.approx(1.0, rel=0.1)

    def test_write_throughput_mbps(self):
        """写吞吐量计算"""
        monitor = ThroughputMonitor(window_seconds=10)
        monitor.record_write(bytes_written=5 * 1024 * 1024)
        assert monitor.write_throughput_mbps == pytest.approx(0.5, rel=0.2)

    def test_ops_per_second(self):
        """操作数秒"""
        monitor = ThroughputMonitor(window_seconds=10)
        for _ in range(20):
            monitor.record_read(ops=1)
        assert monitor.read_ops_per_second == pytest.approx(2.0, rel=0.2)

    def test_prune_old_snapshots(self):
        """旧快照清理"""
        monitor = ThroughputMonitor(window_seconds=0.01)  # 非常短的窗口
        monitor.record_read(bytes_read=1024)
        time.sleep(0.02)  # 等待过期
        # 新记录触发清理
        monitor.record_write(bytes_written=512)
        assert monitor._total_bytes_read == 0  # 旧数据被清除

    def test_get_summary(self):
        """摘要输出"""
        monitor = ThroughputMonitor(window_seconds=60)
        monitor.record_read(bytes_read=1024 * 1024, records=100)
        summary = monitor.get_summary()
        assert "read_throughput_mbps" in summary
        assert "write_throughput_mbps" in summary
        assert summary["total_bytes_read"] == 1024 * 1024

    def test_reset(self):
        """重置"""
        monitor = ThroughputMonitor(window_seconds=60)
        monitor.record_read(bytes_read=1024 * 1024)
        monitor.reset()
        assert monitor._total_bytes_read == 0
        assert len(monitor._snapshots) == 0


# ============================================================================
#  PerformanceReport 测试
# ============================================================================

class TestPerformanceReport:
    """PerformanceReport 性能报告测试"""

    def test_combined_hit_rate(self):
        """综合命中率"""
        report = PerformanceReport(
            cache_levels=[
                CacheLevelStats("L1", hits=8, misses=2),
                CacheLevelStats("L2", hits=5, misses=5),
            ]
        )
        # (8+5)/(8+2+5+5) = 13/20 = 0.65
        assert report.combined_hit_rate == 0.65

    def test_combined_hit_rate_empty(self):
        """空缓存"""
        report = PerformanceReport()
        assert report.combined_hit_rate == 0.0

    def test_timer_summary(self):
        """计时摘要"""
        report = PerformanceReport(
            timers=[
                TimerRecord(name="op_a", duration_ms=10.0),
                TimerRecord(name="op_a", duration_ms=20.0),
                TimerRecord(name="op_b", duration_ms=30.0),
            ]
        )
        summary = report.timer_summary
        assert summary["op_a"] == 15.0
        assert summary["op_b"] == 30.0

    def test_to_dict(self):
        """to_dict"""
        report = PerformanceReport(
            timers=[TimerRecord(name="t", duration_ms=5.0)],
            cache_levels=[CacheLevelStats("L1", hits=1)],
            throughput_mbps=2.5,
        )
        d = report.to_dict()
        assert d["throughput_mbps"] == 2.5
        assert d["combined_hit_rate"] == 1.0


# ============================================================================
#  generate_performance_report 测试
# ============================================================================

class TestGeneratePerformanceReport:
    """generate_performance_report 集成函数测试"""

    def test_basic(self):
        """基本报告生成"""
        tracker = CacheStatsTracker()
        tracker.register_level("L1")
        tracker.record_hit("L1")

        with BenchmarkTimer("test_gen"):
            pass

        report = generate_performance_report(cache_tracker=tracker)
        assert isinstance(report, PerformanceReport)
        assert len(report.timers) >= 1
        assert len(report.cache_levels) >= 1

    def test_without_inputs(self):
        """无输入"""
        report = generate_performance_report()
        assert isinstance(report, PerformanceReport)
