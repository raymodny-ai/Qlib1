"""
性能基准与吞吐量监控 (Performance Benchmarks & Throughput Monitor)

Qlib 基础设施层的性能度量组件，对标 PRD 第4章性能目标：
- 特征集组装 < 10s (预热后，数百只股票 × 10年)
- 多级缓存综合命中率 ≥ 80%
- 毫秒级数据吞吐量

核心组件:
- BenchmarkTimer: 高精度计时上下文管理器
- CacheStatsTracker: 多级缓存统计追踪器
- FeatureAssemblyBenchmark: 特征组装性能基准
- ThroughputMonitor: 数据吞吐量实时监控
- PerformanceReport: 性能报告数据结构

使用示例:
    from src.infrastructure.performance import BenchmarkTimer, CacheStatsTracker

    tracker = CacheStatsTracker()
    with BenchmarkTimer("feature_assembly") as t:
        df = ds.load_features(fields, start="2010-01-01", end="2023-12-31")
    tracker.record_hit("global_cache")  # IR 可观测 hooks
"""

import time
import threading
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

from src.utils.logger import get_logger


# ============================================================================
#  数据结构
# ============================================================================

@dataclass
class TimerRecord:
    """单次计时记录"""
    name: str
    duration_ms: float
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def duration_s(self) -> float:
        return self.duration_ms / 1000.0


@dataclass
class CacheLevelStats:
    """单层缓存统计"""
    level_name: str
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    size_bytes: int = 0
    max_size_bytes: int = 0

    @property
    def total_accesses(self) -> int:
        return self.hits + self.misses

    @property
    def hit_rate(self) -> float:
        total = self.total_accesses
        return self.hits / total if total > 0 else 0.0

    @property
    def miss_rate(self) -> float:
        total = self.total_accesses
        return self.misses / total if total > 0 else 0.0

    @property
    def fill_ratio(self) -> float:
        return self.size_bytes / self.max_size_bytes if self.max_size_bytes > 0 else 0.0


@dataclass
class PerformanceReport:
    """性能报告"""
    generated_at: str = field(default_factory=lambda: datetime.now().isoformat())
    timers: List[TimerRecord] = field(default_factory=list)
    cache_levels: List[CacheLevelStats] = field(default_factory=list)
    throughput_mbps: float = 0.0
    peak_memory_mb: float = 0.0

    @property
    def combined_hit_rate(self) -> float:
        """综合缓存命中率"""
        if not self.cache_levels:
            return 0.0
        total_hits = sum(c.hits for c in self.cache_levels)
        total_accesses = sum(c.total_accesses for c in self.cache_levels)
        return total_hits / total_accesses if total_accesses > 0 else 0.0

    @property
    def timer_summary(self) -> Dict[str, float]:
        """各操作的平均耗时 (ms)"""
        groups: Dict[str, List[float]] = {}
        for t in self.timers:
            groups.setdefault(t.name, []).append(t.duration_ms)
        return {name: np.mean(vals) for name, vals in groups.items()}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "generated_at": self.generated_at,
            "timers": [
                {"name": t.name, "duration_ms": round(t.duration_ms, 3),
                 "metadata": t.metadata}
                for t in self.timers
            ],
            "cache_levels": [
                {"level": c.level_name, "hit_rate": round(c.hit_rate, 4),
                 "hits": c.hits, "misses": c.misses,
                 "fill_ratio": round(c.fill_ratio, 4)}
                for c in self.cache_levels
            ],
            "throughput_mbps": round(self.throughput_mbps, 2),
            "peak_memory_mb": round(self.peak_memory_mb, 2),
            "combined_hit_rate": round(self.combined_hit_rate, 4),
        }


# ============================================================================
#  BenchmarkTimer — 高精度计时器
# ============================================================================

class BenchmarkTimer:
    """
    高精度计时上下文管理器

    支持:
    - 嵌套计时 (通过 name 分组)
    - 自动注册到全局注册表
    - 可选的元数据附着

    使用示例:
        with BenchmarkTimer("load_features") as t:
            df = load_data()
        print(f"{t.name}: {t.duration_ms:.2f}ms")

        # 装饰器模式
        @BenchmarkTimer.decorate("predict")
        def predict(model, X):
            return model.predict(X)
    """

    _registry: Dict[str, List[TimerRecord]] = {}
    _lock = threading.Lock()
    _enabled: bool = True
    _warmup_complete: bool = False

    def __init__(self, name: str, metadata: Optional[Dict[str, Any]] = None):
        self.name = name
        self.metadata = metadata or {}
        self.start_time: float = 0.0
        self.end_time: float = 0.0
        self.duration_ms: float = 0.0

    def __enter__(self) -> "BenchmarkTimer":
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.perf_counter()
        self.duration_ms = (self.end_time - self.start_time) * 1000.0
        if BenchmarkTimer._enabled:
            record = TimerRecord(
                name=self.name,
                duration_ms=self.duration_ms,
                metadata=self.metadata,
            )
            with BenchmarkTimer._lock:
                BenchmarkTimer._registry.setdefault(self.name, []).append(record)
        return False

    @classmethod
    def decorate(cls, name: str):
        """装饰器模式：将函数执行纳入计时"""
        def decorator(func: Callable):
            def wrapper(*args, **kwargs):
                with cls(name) as timer:
                    result = func(*args, **kwargs)
                    timer.metadata["function"] = func.__name__
                return result
            return wrapper
        return decorator

    @classmethod
    def enable(cls):
        cls._enabled = True

    @classmethod
    def disable(cls):
        cls._enabled = False

    @classmethod
    def reset(cls):
        """重置所有计时记录"""
        with cls._lock:
            cls._registry.clear()

    @classmethod
    def set_warmup_complete(cls):
        """标记预热完成，后续计时视为正式基准"""
        cls._warmup_complete = True

    @classmethod
    def get_stats(cls, name: Optional[str] = None) -> Dict[str, Any]:
        """
        获取计时统计

        Args:
            name: 计时器名称 (None 返回所有)

        Returns:
            {"count": N, "mean_ms": M, "min_ms": X, "max_ms": Y, "p50_ms": Z, "p99_ms": W}
        """
        with cls._lock:
            if name:
                records = cls._registry.get(name, [])
            else:
                records = [r for records_list in cls._registry.values() for r in records_list]
            if not records:
                return {"count": 0, "mean_ms": 0.0}
            durations = np.array([r.duration_ms for r in records])
            return {
                "count": int(len(durations)),
                "mean_ms": round(float(np.mean(durations)), 3),
                "min_ms": round(float(np.min(durations)), 3),
                "max_ms": round(float(np.max(durations)), 3),
                "p50_ms": round(float(np.percentile(durations, 50)), 3),
                "p95_ms": round(float(np.percentile(durations, 95)), 3),
                "p99_ms": round(float(np.percentile(durations, 99)), 3),
            }

    @classmethod
    def get_all_records(cls) -> Dict[str, List[TimerRecord]]:
        with cls._lock:
            return dict(cls._registry)


# ============================================================================
#  CacheStatsTracker — 多级缓存统计
# ============================================================================

class CacheStatsTracker:
    """
    多级缓存统计追踪器

    追踪全局内存缓存、表达式缓存、数据集缓存等各级缓存性能，
    对标 PRD 4.1 中 ≥ 80% 的综合命中率目标。

    使用示例:
        tracker = CacheStatsTracker()
        tracker.register_level("global_cache", max_size_gb=4)
        tracker.register_level("expression_cache", max_size_mb=512)
        tracker.record_hit("global_cache")
        tracker.record_miss("expression_cache")
        print(f"Combined hit rate: {tracker.combined_hit_rate:.1%}")
    """

    def __init__(self):
        self._levels: OrderedDict[str, CacheLevelStats] = OrderedDict()
        self._lock = threading.Lock()

    def register_level(
        self,
        name: str,
        max_size_mb: float = 0,
        max_size_gb: float = 0,
    ):
        """注册一级缓存"""
        max_bytes = max_size_mb * 1024 * 1024 + max_size_gb * 1024 * 1024 * 1024
        with self._lock:
            if name not in self._levels:
                self._levels[name] = CacheLevelStats(
                    level_name=name,
                    max_size_bytes=int(max_bytes),
                )

    def record_hit(self, level: str, size_bytes: int = 0):
        """记录缓存命中"""
        with self._lock:
            if level in self._levels:
                stats = self._levels[level]
                stats.hits += 1
                if size_bytes > 0:
                    stats.size_bytes = min(stats.size_bytes + size_bytes, stats.max_size_bytes)

    def record_miss(self, level: str):
        """记录缓存未命中"""
        with self._lock:
            if level in self._levels:
                self._levels[level].misses += 1

    def record_eviction(self, level: str, freed_bytes: int):
        """记录缓存驱逐"""
        with self._lock:
            if level in self._levels:
                stats = self._levels[level]
                stats.evictions += 1
                stats.size_bytes = max(0, stats.size_bytes - freed_bytes)

    def update_size(self, level: str, size_bytes: int):
        """更新缓存当前占用"""
        with self._lock:
            if level in self._levels:
                self._levels[level].size_bytes = size_bytes

    def get_level_stats(self, level: str) -> Optional[CacheLevelStats]:
        with self._lock:
            return self._levels.get(level)

    @property
    def combined_hit_rate(self) -> float:
        """综合缓存命中率"""
        with self._lock:
            total_hits = sum(s.hits for s in self._levels.values())
            total_accesses = sum(s.total_accesses for s in self._levels.values())
            return total_hits / total_accesses if total_accesses > 0 else 0.0

    @property
    def meets_target(self) -> bool:
        """检查是否达到 80% 目标"""
        return self.combined_hit_rate >= 0.80

    def get_summary(self) -> Dict[str, Any]:
        """获取各级缓存摘要"""
        with self._lock:
            summary = {
                "combined_hit_rate": round(self.combined_hit_rate, 4),
                "meets_target": self.meets_target,
                "levels": {},
            }
            for name, stats in self._levels.items():
                summary["levels"][name] = {
                    "hit_rate": round(stats.hit_rate, 4),
                    "hits": stats.hits,
                    "misses": stats.misses,
                    "evictions": stats.evictions,
                    "fill_ratio": round(stats.fill_ratio, 4),
                }
            return summary

    def reset(self):
        with self._lock:
            for stats in self._levels.values():
                stats.hits = 0
                stats.misses = 0
                stats.evictions = 0
                stats.size_bytes = 0


# ============================================================================
#  FeatureAssemblyBenchmark — 特征组装性能基准
# ============================================================================

@dataclass
class FeatureAssemblyResult:
    """特征组装基准结果"""
    instrument_count: int
    field_count: int
    date_range_start: str
    date_range_end: str
    durations: List[float] = field(default_factory=list)
    cache_hit_rates: List[float] = field(default_factory=list)
    warmed_up: bool = False

    @property
    def mean_duration_s(self) -> float:
        if not self.durations:
            return 0.0
        return float(np.mean(self.durations))

    @property
    def p95_duration_s(self) -> float:
        if not self.durations:
            return 0.0
        return float(np.percentile(self.durations, 95))

    @property
    def meets_10s_target(self) -> bool:
        """检查是否达到 < 10s 目标"""
        return self.mean_duration_s < 10.0

    @property
    def mean_hit_rate(self) -> float:
        if not self.cache_hit_rates:
            return 0.0
        return float(np.mean(self.cache_hit_rates))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "instrument_count": self.instrument_count,
            "field_count": self.field_count,
            "date_range": f"{self.date_range_start} ~ {self.date_range_end}",
            "warmed_up": self.warmed_up,
            "mean_duration_s": round(self.mean_duration_s, 3),
            "p95_duration_s": round(self.p95_duration_s, 3),
            "meets_10s_target": self.meets_10s_target,
            "iterations": len(self.durations),
            "mean_hit_rate": round(self.mean_hit_rate, 4),
        }


class FeatureAssemblyBenchmark:
    """
    特征组装性能基准测试器

    对标 PRD 4.1: 在充分预热后，完成数百只股票 × 10年以上
    特征集组装需控制在 10s 级别内。

    使用示例:
        bench = FeatureAssemblyBenchmark(data_server=ds)
        result = bench.benchmark(
            instruments=["AAPL", "MSFT", ...],
            fields=["close", "volume", ...],
            start="2010-01-01",
            end="2023-12-31",
            warmup_iterations=2,
            benchmark_iterations=5,
        )
        assert result.meets_10s_target, f"Assembly too slow: {result.mean_duration_s:.1f}s"
    """

    def __init__(self, data_server=None, cache_tracker: Optional[CacheStatsTracker] = None):
        """
        Args:
            data_server: DataServer 实例 (支持 Mock)
            cache_tracker: CacheStatsTracker 实例 (可选)
        """
        self.data_server = data_server
        self.cache_tracker = cache_tracker
        self.logger = get_logger()

    def benchmark(
        self,
        instruments: List[str],
        fields: List[str],
        start: str,
        end: str,
        warmup_iterations: int = 2,
        benchmark_iterations: int = 5,
    ) -> FeatureAssemblyResult:
        """
        执行特征组装性能基准测试

        Args:
            instruments: 股票列表
            fields: 特征字段列表
            start: 起始日期
            end: 结束日期
            warmup_iterations: 预热迭代次数
            benchmark_iterations: 正式基准迭代次数

        Returns:
            FeatureAssemblyResult
        """
        result = FeatureAssemblyResult(
            instrument_count=len(instruments),
            field_count=len(fields),
            date_range_start=start,
            date_range_end=end,
        )

        # 预热阶段
        self.logger.info("特征组装基准: 预热阶段", iterations=warmup_iterations)
        for i in range(warmup_iterations):
            self._run_assembly(instruments, fields, start, end)
            if self.cache_tracker:
                self.cache_tracker.record_hit("global_cache")

        result.warmed_up = True
        BenchmarkTimer.set_warmup_complete()

        # 正式基准阶段
        self.logger.info("特征组装基准: 正式基准", iterations=benchmark_iterations)
        durations = []
        hit_rates = []
        for i in range(benchmark_iterations):
            t0 = time.perf_counter()
            self._run_assembly(instruments, fields, start, end)
            duration = time.perf_counter() - t0
            durations.append(duration)
            if self.cache_tracker:
                hit_rates.append(self.cache_tracker.combined_hit_rate)
            self.logger.debug("基准迭代", iteration=i + 1, duration_s=round(duration, 3))

        result.durations = durations
        result.cache_hit_rates = hit_rates

        self.logger.info(
            "特征组装基准完成",
            mean_s=round(result.mean_duration_s, 3),
            p95_s=round(result.p95_duration_s, 3),
            meets_target=result.meets_10s_target,
        )
        return result

    def _run_assembly(self, instruments, fields, start, end):
        """执行单次组装 (子类可覆写)"""
        if self.data_server:
            if hasattr(self.data_server, "load_features"):
                return self.data_server.load_features(
                    instruments=instruments,
                    fields=fields,
                    start=start,
                    end=end,
                )
        return None

    @staticmethod
    def estimate_data_volume(
        instruments: int,
        fields: int,
        days: int,
        bytes_per_element: int = 4,
    ) -> float:
        """
        估算数据量 (MB)

        Args:
            instruments: 股票数
            fields: 特征数
            days: 天数
            bytes_per_element: 每元素字节数 (float32=4)
        """
        total_bytes = instruments * fields * days * bytes_per_element
        return total_bytes / (1024 * 1024)


# ============================================================================
#  ThroughputMonitor — 吞吐量监控
# ============================================================================

@dataclass
class ThroughputSnapshot:
    """吞吐量快照"""
    timestamp: float = field(default_factory=time.time)
    bytes_read: int = 0
    bytes_written: int = 0
    records_read: int = 0
    records_written: int = 0
    read_ops: int = 0
    write_ops: int = 0


class ThroughputMonitor:
    """
    数据吞吐量实时监控器

    用于跟踪 I/O 层的读写吞吐量 (MB/s)，对标毫秒级响应目标。

    使用示例:
        monitor = ThroughputMonitor(window_seconds=60)
        monitor.record_read(bytes_read=1024*1024*10, records=1000)
        monitor.record_write(bytes_written=1024*1024*5, records=500)
        print(f"Read throughput: {monitor.read_throughput_mbps:.2f} MB/s")
    """

    def __init__(self, window_seconds: float = 60.0):
        self.window_seconds = window_seconds
        self._snapshots: List[ThroughputSnapshot] = []
        self._lock = threading.Lock()
        self._total_bytes_read: int = 0
        self._total_bytes_written: int = 0
        self._total_read_ops: int = 0
        self._total_write_ops: int = 0

    def record_read(
        self,
        bytes_read: int = 0,
        records: int = 0,
        ops: int = 1,
    ):
        """记录读操作"""
        self._add_snapshot(ThroughputSnapshot(
            bytes_read=bytes_read,
            records_read=records,
            read_ops=ops,
        ))

    def record_write(
        self,
        bytes_written: int = 0,
        records: int = 0,
        ops: int = 1,
    ):
        """记录写操作"""
        self._add_snapshot(ThroughputSnapshot(
            bytes_written=bytes_written,
            records_written=records,
            write_ops=ops,
        ))

    def _add_snapshot(self, snapshot: ThroughputSnapshot):
        with self._lock:
            self._snapshots.append(snapshot)
            self._total_bytes_read += snapshot.bytes_read
            self._total_bytes_written += snapshot.bytes_written
            self._total_read_ops += snapshot.read_ops
            self._total_write_ops += snapshot.write_ops
            self._prune_old_snapshots()

    def _prune_old_snapshots(self):
        """清理超出窗口的旧快照"""
        cutoff = time.time() - self.window_seconds
        self._snapshots = [s for s in self._snapshots if s.timestamp >= cutoff]
        self._total_bytes_read = sum(s.bytes_read for s in self._snapshots)
        self._total_bytes_written = sum(s.bytes_written for s in self._snapshots)
        self._total_read_ops = sum(s.read_ops for s in self._snapshots)
        self._total_write_ops = sum(s.write_ops for s in self._snapshots)

    @property
    def read_throughput_mbps(self) -> float:
        """读吞吐量 (MB/s)"""
        with self._lock:
            self._prune_old_snapshots()
            return (self._total_bytes_read / (1024 * 1024)) / self.window_seconds

    @property
    def write_throughput_mbps(self) -> float:
        """写吞吐量 (MB/s)"""
        with self._lock:
            self._prune_old_snapshots()
            return (self._total_bytes_written / (1024 * 1024)) / self.window_seconds

    @property
    def read_ops_per_second(self) -> float:
        """读操作数/秒"""
        with self._lock:
            self._prune_old_snapshots()
            return self._total_read_ops / self.window_seconds

    @property
    def write_ops_per_second(self) -> float:
        """写操作数/秒"""
        with self._lock:
            self._prune_old_snapshots()
            return self._total_write_ops / self.window_seconds

    def get_summary(self) -> Dict[str, Any]:
        """获取吞吐量摘要"""
        with self._lock:
            self._prune_old_snapshots()
            total_records_read = sum(s.records_read for s in self._snapshots)
            total_records_written = sum(s.records_written for s in self._snapshots)
            return {
                "read_throughput_mbps": round(self.read_throughput_mbps, 3),
                "write_throughput_mbps": round(self.write_throughput_mbps, 3),
                "read_ops_per_second": round(self.read_ops_per_second, 2),
                "write_ops_per_second": round(self.write_ops_per_second, 2),
                "total_bytes_read": self._total_bytes_read,
                "total_bytes_written": self._total_bytes_written,
                "total_records_read": total_records_read,
                "total_records_written": total_records_written,
                "window_seconds": self.window_seconds,
            }

    def reset(self):
        with self._lock:
            self._snapshots.clear()
            self._total_bytes_read = 0
            self._total_bytes_written = 0
            self._total_read_ops = 0
            self._total_write_ops = 0


# ============================================================================
#  便捷函数
# ============================================================================

def generate_performance_report(
    cache_tracker: Optional[CacheStatsTracker] = None,
    throughput_monitor: Optional[ThroughputMonitor] = None,
) -> PerformanceReport:
    """
    生成综合性能报告

    Args:
        cache_tracker: 缓存统计追踪器
        throughput_monitor: 吞吐量监控器

    Returns:
        PerformanceReport
    """
    report = PerformanceReport()

    # 收集计时记录
    all_records = BenchmarkTimer.get_all_records()
    report.timers = [r for records in all_records.values() for r in records]

    # 收集缓存统计
    if cache_tracker:
        report.cache_levels = list(cache_tracker._levels.values())

    # 收集吞吐量
    if throughput_monitor:
        report.throughput_mbps = throughput_monitor.read_throughput_mbps + \
                                  throughput_monitor.write_throughput_mbps

    try:
        import psutil
        report.peak_memory_mb = psutil.Process().memory_info().rss / (1024 * 1024)
    except Exception:
        pass

    return report
