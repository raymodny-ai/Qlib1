"""
缓存监控器 (Cache Monitor)

封装 CacheStatsTracker 为面向运维的生产级监控层，
提供三层缓存命中率采集与 Prometheus 兼容指标输出。

PRD 第4章: 多级缓存综合命中率 ≥ 80%。

缓存层级:
- Global Cache (L1): 特征文件级, DataServer 的 OrderedDict LRU
- Expression Cache (L2): 表达式计算结果, ExpressionEngine
- Dataset Cache (L3): 数据集查询结果, 时序对齐后的 DataFrame

使用示例:
    from src.infrastructure.cache_monitor import CacheMonitor

    monitor = CacheMonitor()
    print(monitor.health_check())  # {"status": "healthy", "combined_hit_rate": 0.92}
    print(monitor.prometheus_metrics())  # Prometheus text format
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional

from src.utils.logger import get_logger


@dataclass
class CacheHealth:
    """缓存健康状态"""
    status: str  # "healthy" | "degraded" | "critical"
    combined_hit_rate: float
    global_hit_rate: float
    expression_hit_rate: float
    dataset_hit_rate: float
    total_entries: int
    total_evictions: int
    recommendations: list


class CacheMonitor:
    """
    三层缓存监控器

    封装 CacheStatsTracker (src.infrastructure.performance)，
    提供面向运维的健康检查和 Prometheus 指标输出。

    健康阈值:
    - 综合命中率 < 50% → critical
    - 综合命中率 < 80% → degraded
    - 综合命中率 ≥ 80% → healthy
    """

    HEALTHY_THRESHOLD = 0.80
    DEGRADED_THRESHOLD = 0.50

    def __init__(self):
        self.logger = get_logger()
        self._tracker = None  # 惰性初始化

    def _get_tracker(self):
        """获取 CacheStatsTracker 单例"""
        if self._tracker is None:
            from src.infrastructure.performance import CacheStatsTracker
            self._tracker = CacheStatsTracker.get_instance()
        return self._tracker

    def get_global_stats(self) -> Dict[str, Any]:
        """获取 Global Cache (L1) 统计"""
        tracker = self._get_tracker()
        try:
            summary = tracker.get_summary()
            return {
                "level": "global_cache",
                "hit_rate": summary.get("global_hit_rate", 0.0),
                "hits": summary.get("global_hits", 0),
                "misses": summary.get("global_misses", 0),
                "size": summary.get("global_cache_size", 0),
                "max_size": summary.get("global_cache_max", 0),
            }
        except Exception:
            return {"level": "global_cache", "hit_rate": 0.0, "hits": 0, "misses": 0}

    def get_expression_stats(self) -> Dict[str, Any]:
        """获取 Expression Cache (L2) 统计"""
        tracker = self._get_tracker()
        try:
            summary = tracker.get_summary()
            return {
                "level": "expression_cache",
                "hit_rate": summary.get("expression_hit_rate", 0.0),
                "hits": summary.get("expression_hits", 0),
                "misses": summary.get("expression_misses", 0),
                "size": summary.get("expression_cache_size", 0),
            }
        except Exception:
            return {"level": "expression_cache", "hit_rate": 0.0, "hits": 0, "misses": 0}

    def get_dataset_stats(self) -> Dict[str, Any]:
        """获取 Dataset Cache (L3) 统计"""
        tracker = self._get_tracker()
        try:
            summary = tracker.get_summary()
            return {
                "level": "dataset_cache",
                "hit_rate": summary.get("dataset_hit_rate", 0.0),
                "hits": summary.get("dataset_hits", 0),
                "misses": summary.get("dataset_misses", 0),
                "size": summary.get("dataset_cache_size", 0),
            }
        except Exception:
            return {"level": "dataset_cache", "hit_rate": 0.0, "hits": 0, "misses": 0}

    def get_combined_hit_rate(self) -> float:
        """获取三层综合命中率"""
        tracker = self._get_tracker()
        try:
            return tracker.combined_hit_rate()
        except Exception:
            return 0.0

    def health_check(self) -> CacheHealth:
        """
        缓存健康检查

        Returns:
            CacheHealth: 包含健康状态和建议
        """
        combined_rate = self.get_combined_hit_rate()
        global_stats = self.get_global_stats()
        expr_stats = self.get_expression_stats()
        ds_stats = self.get_dataset_stats()

        evictions = (
            global_stats.get("evictions", 0) +
            expr_stats.get("evictions", 0) +
            ds_stats.get("evictions", 0)
        )

        total_entries = (
            global_stats.get("size", 0) +
            expr_stats.get("size", 0) +
            ds_stats.get("size", 0)
        )

        # 判断健康状态
        if combined_rate >= self.HEALTHY_THRESHOLD:
            status = "healthy"
            recommendations = ["缓存命中率正常，无需调整"]
        elif combined_rate >= self.DEGRADED_THRESHOLD:
            status = "degraded"
            recommendations = [
                "建议增加缓存容量 (当前综合命中率低于 80%)",
                "考虑对热点特征集进行预热",
                "检查是否存在频繁的缓存驱逐",
            ]
        else:
            status = "critical"
            recommendations = [
                "缓存命中率严重不足 (综合 < 50%)",
                "建议立即增加缓存容量并分析未命中模式",
                "检查 DataServer 是否正常运行",
                "考虑降低特征计算复杂度或增加预计算",
            ]

        return CacheHealth(
            status=status,
            combined_hit_rate=round(combined_rate, 4),
            global_hit_rate=round(global_stats["hit_rate"], 4),
            expression_hit_rate=round(expr_stats["hit_rate"], 4),
            dataset_hit_rate=round(ds_stats["hit_rate"], 4),
            total_entries=total_entries,
            total_evictions=evictions,
            recommendations=recommendations,
        )

    def prometheus_metrics(self) -> str:
        """
        输出 Prometheus 兼容格式的文本指标

        供 /api/v1/metrics 端点或 Prometheus scrape 使用。

        Returns:
            Prometheus 文本格式字符串
        """
        health = self.health_check()
        global_s = self.get_global_stats()
        expr_s = self.get_expression_stats()
        ds_s = self.get_dataset_stats()

        lines = [
            "# HELP qlib_cache_hit_rate Cache hit rate by level",
            "# TYPE qlib_cache_hit_rate gauge",
            f'qlib_cache_hit_rate{{level="global"}} {global_s["hit_rate"]}',
            f'qlib_cache_hit_rate{{level="expression"}} {expr_s["hit_rate"]}',
            f'qlib_cache_hit_rate{{level="dataset"}} {ds_s["hit_rate"]}',
            f'qlib_cache_hit_rate{{level="combined"}} {health.combined_hit_rate}',
            "",
            "# HELP qlib_cache_hits_total Total cache hits by level",
            "# TYPE qlib_cache_hits_total counter",
            f'qlib_cache_hits_total{{level="global"}} {global_s["hits"]}',
            f'qlib_cache_hits_total{{level="expression"}} {expr_s["hits"]}',
            f'qlib_cache_hits_total{{level="dataset"}} {ds_s["hits"]}',
            "",
            "# HELP qlib_cache_misses_total Total cache misses by level",
            "# TYPE qlib_cache_misses_total counter",
            f'qlib_cache_misses_total{{level="global"}} {global_s["misses"]}',
            f'qlib_cache_misses_total{{level="expression"}} {expr_s["misses"]}',
            f'qlib_cache_misses_total{{level="dataset"}} {ds_s["misses"]}',
            "",
            "# HELP qlib_cache_health_status Cache health status",
            "# TYPE qlib_cache_health_status gauge",
            f'qlib_cache_health_status{{status="{health.status}"}} 1',
            "",
            "# HELP qlib_cache_evictions_total Total cache evictions",
            "# TYPE qlib_cache_evictions_total counter",
            f"qlib_cache_evictions_total {health.total_evictions}",
            "",
        ]
        return "\n".join(lines)

    def log_summary(self):
        """将缓存摘要写入日志"""
        health = self.health_check()
        self.logger.info(
            "缓存健康检查",
            status=health.status,
            combined_hit_rate=f"{health.combined_hit_rate:.2%}",
            global_hit_rate=f"{health.global_hit_rate:.2%}",
            expression_hit_rate=f"{health.expression_hit_rate:.2%}",
            dataset_hit_rate=f"{health.dataset_hit_rate:.2%}",
            entries=health.total_entries,
            evictions=health.total_evictions,
        )
        if health.recommendations:
            for rec in health.recommendations:
                self.logger.info(f"  建议: {rec}")
