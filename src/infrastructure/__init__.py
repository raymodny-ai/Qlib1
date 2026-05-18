"""
基础设施层 (Infrastructure Layer)

Qlib 四层架构的底层底座，负责 I/O 吞吐、计算调度与系统监控。

核心组件:
- DataServer: 高速二进制特征池与 PIT 数据库管理
- TrainerDispatcher: 多 GPU 训练任务编排
- HealthChecker: 自动化健康探针与故障告警
- Performance: 性能基准与吞吐量监控
- CircuitBreaker: 熔断器与自愈机制
"""

from src.infrastructure.data_server import (
    DataServer,
    MemoryCache,
    BinFileRegistry,
    BinFileMeta,
)

from src.infrastructure.trainer import (
    TrainerDispatcher,
    GPUPool,
    GPUDevice,
    CheckpointManager,
    Checkpoint,
    EarlyStopping,
    TrainingResult,
)

from src.infrastructure.health_checker import (
    HealthChecker,
    HealthReport,
    CheckResult,
    CheckStatus,
    PITMonotonicityValidator,
    GapDetector,
    SystemMonitor,
)

from src.infrastructure.signal_exporter import (
    SignalExporter,
    SignalBatch,
    SignalEntry,
    OMSAdapter,
    Order,
    ProductionGateway,
)

from src.infrastructure.performance import (
    BenchmarkTimer,
    CacheStatsTracker,
    FeatureAssemblyBenchmark,
    FeatureAssemblyResult,
    ThroughputMonitor,
    PerformanceReport,
    TimerRecord,
    CacheLevelStats,
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
    HealingResult,
    with_resilience,
)

__all__ = [
    # DataServer
    "DataServer",
    "MemoryCache",
    "BinFileRegistry",
    "BinFileMeta",
    # Trainer
    "TrainerDispatcher",
    "GPUPool",
    "GPUDevice",
    "CheckpointManager",
    "Checkpoint",
    "EarlyStopping",
    "TrainingResult",
    # Health
    "HealthChecker",
    "HealthReport",
    "CheckResult",
    "CheckStatus",
    "PITMonotonicityValidator",
    "GapDetector",
    "SystemMonitor",
    # Signal
    "SignalExporter",
    "SignalBatch",
    "SignalEntry",
    "OMSAdapter",
    "Order",
    "ProductionGateway",
    # Performance
    "BenchmarkTimer",
    "CacheStatsTracker",
    "FeatureAssemblyBenchmark",
    "FeatureAssemblyResult",
    "ThroughputMonitor",
    "PerformanceReport",
    "TimerRecord",
    "CacheLevelStats",
    "generate_performance_report",
    # Circuit Breaker & Resilience
    "CircuitBreaker",
    "CircuitState",
    "CircuitBreakerOpenError",
    "RetryPolicy",
    "SlaTracker",
    "AutoHealingManager",
    "HealingAction",
    "HealingResult",
    "with_resilience",
]
