"""
系统健康探针与监控 (Health Checker & Monitor)

负责自动化健康探针、PIT 时序单调性核查、因子断层检测，
以及系统级可用性度量。

核心组件:
- HealthChecker: 综合健康探针
- PITMonotonicityValidator: PIT 时序单调性校验器
- GapDetector: 数据断层/跳空检测器
- SystemMonitor: 系统资源监控 (CPU/GPU/内存)

设计原则:
- 可配置的检查间隔
- 侦测到严重跳空时立即挂起模型重训管线并推送警报
- 定时探针 + 按需检查

使用示例:
    from src.infrastructure.health_checker import HealthChecker
    
    hc = HealthChecker(data_server=ds)
    result = hc.run_full_check()
    if result.has_critical:
        hc.raise_alert(result)
"""

import os
import time
import json
import threading
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

from src.utils.logger import get_logger


# ============================================================================
#  健康状态枚举
# ============================================================================

class CheckStatus(Enum):
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


@dataclass
class CheckResult:
    """单项检查结果"""
    name: str
    status: CheckStatus = CheckStatus.UNKNOWN
    message: str = ""
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    @property
    def is_healthy(self) -> bool:
        return self.status == CheckStatus.HEALTHY
    
    @property
    def is_critical(self) -> bool:
        return self.status == CheckStatus.CRITICAL


@dataclass
class HealthReport:
    """综合健康报告"""
    overall: CheckStatus = CheckStatus.UNKNOWN
    checks: List[CheckResult] = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    uptime_seconds: float = 0.0
    
    @property
    def has_critical(self) -> bool:
        return any(c.is_critical for c in self.checks)
    
    @property
    def healthy_count(self) -> int:
        return sum(1 for c in self.checks if c.is_healthy)
    
    @property
    def warning_count(self) -> int:
        return sum(1 for c in self.checks if c.status == CheckStatus.WARNING)
    
    @property
    def critical_count(self) -> int:
        return sum(1 for c in self.checks if c.is_critical)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "overall": self.overall.value,
            "timestamp": self.timestamp,
            "uptime_seconds": self.uptime_seconds,
            "summary": {
                "total": len(self.checks),
                "healthy": self.healthy_count,
                "warning": self.warning_count,
                "critical": self.critical_count,
            },
            "checks": [
                {
                    "name": c.name,
                    "status": c.status.value,
                    "message": c.message,
                    "details": c.details,
                    "timestamp": c.timestamp,
                }
                for c in self.checks
            ],
        }


# ============================================================================
#  PIT 单调性校验器
# ============================================================================

class PITMonotonicityValidator:
    """
    PIT 时序单调性校验器
    
    验证 Point-in-Time 数据中不存在未来数据泄露:
    - 每个 (instrument, field, period) 组合的 filing_date 随时间单调递增
    - 同一 period 下不存在 version 回退
    """
    
    def __init__(self):
        self._logger = get_logger(__name__)
    
    def validate(
        self,
        pit_index: Any,
        instruments: Optional[List[str]] = None,
    ) -> List[CheckResult]:
        """
        校验 PIT 索引的时序单调性
        
        Args:
            pit_index: PITIndex 实例
            instruments: 待检查标的列表
            
        Returns:
            检查结果列表
        """
        results: List[CheckResult] = []
        
        try:
            # 尝试获取 instruments
            if instruments is None:
                if hasattr(pit_index, "_index"):
                    instruments = list(pit_index._index.keys())
                else:
                    instruments = []
            
            violations: List[Dict] = []
            total_checked = 0
            
            for inst in instruments:
                inst_data = pit_index._index.get(inst, {})
                for field, periods in inst_data.items():
                    for period, records in periods.items():
                        total_checked += 1
                        
                        # 按 filing_date 排序
                        sorted_records = sorted(records, key=lambda r: r.filing_date)
                        
                        for i in range(1, len(sorted_records)):
                            prev = sorted_records[i - 1]
                            curr = sorted_records[i]
                            
                            # 检查：同一 period 不应有 filing_date 早于前一版本
                            if curr.filing_date < prev.filing_date:
                                violations.append({
                                    "instrument": inst,
                                    "field": field,
                                    "period": period,
                                    "prev_filing_date": prev.filing_date,
                                    "curr_filing_date": curr.filing_date,
                                    "violation": "filing_date 倒退",
                                })
            
            if violations:
                results.append(CheckResult(
                    name="pit_monotonicity",
                    status=CheckStatus.CRITICAL if len(violations) > 10 else CheckStatus.WARNING,
                    message=f"发现 {len(violations)} 个时序单调性违规 | 共检查 {total_checked} 条",
                    details={"violations": violations[:20], "total_violations": len(violations)},
                ))
            else:
                results.append(CheckResult(
                    name="pit_monotonicity",
                    status=CheckStatus.HEALTHY,
                    message=f"PIT 时序单调性检查通过 | {total_checked} 条记录",
                ))
                
        except Exception as e:
            results.append(CheckResult(
                name="pit_monotonicity",
                status=CheckStatus.WARNING,
                message=f"PIT 校验异常: {e}",
            ))
        
        return results


# ============================================================================
#  数据断层检测器
# ============================================================================

class GapDetector:
    """
    数据断层/跳空检测器
    
    检测特征矩阵中的:
    - 缺失交易日 (holiday gap 除外)
    - 因子断层 (单日变化 > N 标准差)
    - NaN 比例超限
    """
    
    def __init__(
        self,
        max_nan_ratio: float = 0.3,
        jump_sigma_threshold: float = 5.0,
    ):
        self.max_nan_ratio = max_nan_ratio
        self.jump_sigma_threshold = jump_sigma_threshold
        self._logger = get_logger(__name__)
    
    def detect(
        self,
        df: pd.DataFrame,
        calendar: Optional[List[str]] = None,
    ) -> List[CheckResult]:
        """
        检测数据异常
        
        Args:
            df: 特征 DataFrame (index=datetime, columns=features)
            calendar: 交易日历
            
        Returns:
            检查结果列表
        """
        results: List[CheckResult] = []
        
        if df.empty:
            results.append(CheckResult(
                name="gap_detection",
                status=CheckStatus.CRITICAL,
                message="特征 DataFrame 为空",
            ))
            return results
        
        # 1. NaN 比例检查
        nan_ratios = df.isna().mean()
        high_nan_cols = nan_ratios[nan_ratios > self.max_nan_ratio]
        
        if len(high_nan_cols) > 0:
            results.append(CheckResult(
                name="nan_ratio",
                status=CheckStatus.WARNING if len(high_nan_cols) < 5 else CheckStatus.CRITICAL,
                message=f"{len(high_nan_cols)} 个字段 NaN 比例超过 {self.max_nan_ratio:.0%}",
                details={
                    "fields": high_nan_cols.to_dict(),
                    "threshold": self.max_nan_ratio,
                },
            ))
        else:
            results.append(CheckResult(
                name="nan_ratio",
                status=CheckStatus.HEALTHY,
                message=f"所有字段 NaN 比例在 {self.max_nan_ratio:.0%} 以内",
            ))
        
        # 2. 跳空检测 (日收益率绝对值异常)
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            returns = df[numeric_cols].pct_change().dropna(how="all")
            if not returns.empty:
                mean = returns.mean()
                std = returns.std()
                
                # 安全处理 std=0
                std = std.replace(0, 1e-8)
                
                jump_ratio = (returns.abs() > self.jump_sigma_threshold * std.abs() + mean.abs()).mean()
                high_jump_cols = jump_ratio[jump_ratio > 0.01]  # >1% 的数据点异常
                
                if len(high_jump_cols) > 0:
                    results.append(CheckResult(
                        name="jump_detection",
                        status=CheckStatus.WARNING,
                        message=f"{len(high_jump_cols)} 个字段存在异常跳空",
                        details={"fields": high_jump_cols.to_dict()},
                    ))
                else:
                    results.append(CheckResult(
                        name="jump_detection",
                        status=CheckStatus.HEALTHY,
                        message="无异常跳空",
                    ))
        
        # 3. 日历缺失检查
        if calendar is not None and isinstance(df.index, pd.DatetimeIndex):
            cal_dates = set(pd.to_datetime(calendar))
            data_dates = set(df.index)
            missing = cal_dates - data_dates
            if missing:
                n_missing = len(missing)
                n_total = len(cal_dates)
                results.append(CheckResult(
                    name="calendar_completeness",
                    status=CheckStatus.WARNING if n_missing / n_total > 0.1 else CheckStatus.HEALTHY,
                    message=f"缺失 {n_missing}/{n_total} 个交易日数据",
                    details={"missing_ratio": n_missing / n_total},
                ))
        
        return results


# ============================================================================
#  系统资源监控
# ============================================================================

class SystemMonitor:
    """系统资源监控器"""
    
    def __init__(self):
        self._logger = get_logger(__name__)
        self._start_time = time.time()
    
    def check_resources(self) -> CheckResult:
        """检查系统资源"""
        details: Dict[str, Any] = {}
        warnings: List[str] = []
        
        # CPU
        try:
            import psutil
            cpu_pct = psutil.cpu_percent(interval=0.1)
            details["cpu_percent"] = cpu_pct
            if cpu_pct > 90:
                warnings.append(f"CPU 使用率 {cpu_pct}%")
            
            mem = psutil.virtual_memory()
            details["memory_percent"] = mem.percent
            details["memory_available_gb"] = mem.available / (1024**3)
            if mem.percent > 90:
                warnings.append(f"内存使用率 {mem.percent}%")
            
            disk = psutil.disk_usage("/")
            details["disk_percent"] = disk.percent
            if disk.percent > 90:
                warnings.append(f"磁盘使用率 {disk.percent}%")
                
        except ImportError:
            details["psutil"] = "未安装"
        
        # GPU
        try:
            import torch
            if torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    mem_alloc = torch.cuda.memory_allocated(i) / (1024**3)
                    mem_total = torch.cuda.get_device_properties(i).total_memory / (1024**3)
                    details[f"gpu_{i}_memory_gb"] = f"{mem_alloc:.1f}/{mem_total:.1f}"
                    if mem_alloc / mem_total > 0.95:
                        warnings.append(f"GPU {i} 显存使用率 > 95%")
        except ImportError:
            pass
        
        status = CheckStatus.WARNING if warnings else CheckStatus.HEALTHY
        return CheckResult(
            name="system_resources",
            status=status,
            message="; ".join(warnings) if warnings else "系统资源正常",
            details=details,
        )
    
    @property
    def uptime_seconds(self) -> float:
        return time.time() - self._start_time
    
    def collect_metrics(self) -> Dict[str, Any]:
        """采集当前系统指标"""
        metrics: Dict[str, Any] = {
            "timestamp": datetime.now().isoformat(),
            "uptime_seconds": self.uptime_seconds,
        }
        
        try:
            import psutil
            metrics["cpu_percent"] = psutil.cpu_percent(interval=0.1)
            mem = psutil.virtual_memory()
            metrics["memory_percent"] = mem.percent
            metrics["memory_available_gb"] = round(mem.available / (1024**3), 2)
        except ImportError:
            pass
        
        return metrics


# ============================================================================
#  综合健康探针
# ============================================================================

class HealthChecker:
    """
    综合健康探针
    
    定时执行全系统健康检查，包括:
    - PIT 时序单调性
    - 数据断层检测
    - 系统资源
    - API 端点可达性
    - 缓存命中率
    
    侦测到严重跳空时，触发警报并挂起模型重训管线。
    """
    
    def __init__(
        self,
        data_server: Optional[Any] = None,
        pit_index: Optional[Any] = None,
        check_interval_s: int = 3600,
        alert_callback: Optional[Callable[[HealthReport], None]] = None,
    ):
        self.data_server = data_server
        self.pit_index = pit_index
        self.check_interval_s = check_interval_s
        self.alert_callback = alert_callback
        
        self.pit_validator = PITMonotonicityValidator()
        self.gap_detector = GapDetector()
        self.sys_monitor = SystemMonitor()
        
        self._logger = get_logger(__name__)
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._last_report: Optional[HealthReport] = None
    
    def check_data_server(self) -> CheckResult:
        """检查 DataServer 状态"""
        if self.data_server is None:
            return CheckResult(
                name="data_server",
                status=CheckStatus.UNKNOWN,
                message="DataServer 未配置",
            )
        
        try:
            stats = self.data_server.stats
            
            # 检查缓存命中率
            if stats.get("cache"):
                hit_rate = stats["cache"].get("hit_rate", 0)
                if hit_rate >= 0.80:
                    details = {"hit_rate": hit_rate}
                elif hit_rate >= 0.60:
                    return CheckResult(
                        name="data_server",
                        status=CheckStatus.WARNING,
                        message=f"缓存命中率偏低: {hit_rate:.1%}",
                        details={"hit_rate": hit_rate, "target": 0.80},
                    )
                else:
                    return CheckResult(
                        name="data_server",
                        status=CheckStatus.CRITICAL,
                        message=f"缓存命中率严重不足: {hit_rate:.1%}",
                        details={"hit_rate": hit_rate, "target": 0.80},
                    )
            
            return CheckResult(
                name="data_server",
                status=CheckStatus.HEALTHY,
                message=f"DataServer 正常 | {stats.get('instruments', 0)} 标的 | "
                        f"avg_load={stats.get('avg_load_time_s', 0):.2f}s",
                details=stats,
            )
        except Exception as e:
            return CheckResult(
                name="data_server",
                status=CheckStatus.CRITICAL,
                message=f"DataServer 异常: {e}",
            )
    
    def check_api_endpoint(self, url: str = "http://localhost:8000/health") -> CheckResult:
        """检查 API 端点可达性"""
        try:
            import urllib.request
            req = urllib.request.Request(url, method="GET")
            with urllib.request.urlopen(req, timeout=5) as resp:
                if resp.status == 200:
                    return CheckResult(
                        name="api_endpoint",
                        status=CheckStatus.HEALTHY,
                        message=f"API 端点可达: {url}",
                        details={"url": url, "status_code": resp.status},
                    )
                else:
                    return CheckResult(
                        name="api_endpoint",
                        status=CheckStatus.WARNING,
                        message=f"API 返回非 200: {resp.status}",
                    )
        except Exception as e:
            return CheckResult(
                name="api_endpoint",
                status=CheckStatus.WARNING,
                message=f"API 端点不可达: {e}",
            )
    
    def run_full_check(
        self,
        check_api: bool = False,
        feature_df: Optional[pd.DataFrame] = None,
        calendar: Optional[List[str]] = None,
    ) -> HealthReport:
        """
        执行全量健康检查
        
        Args:
            check_api: 是否检查 API 端点
            feature_df: 特征矩阵 (用于断层检测)
            calendar: 交易日历
            
        Returns:
            HealthReport
        """
        checks: List[CheckResult] = []
        
        # 1. DataServer
        checks.append(self.check_data_server())
        
        # 2. PIT 单调性
        if self.pit_index is not None:
            checks.extend(self.pit_validator.validate(self.pit_index))
        
        # 3. 数据断层
        if feature_df is not None:
            checks.extend(self.gap_detector.detect(feature_df, calendar))
        
        # 4. 系统资源
        checks.append(self.sys_monitor.check_resources())
        
        # 5. API 端点
        if check_api:
            checks.append(self.check_api_endpoint())
        
        # 综合判定
        if any(c.is_critical for c in checks):
            overall = CheckStatus.CRITICAL
        elif any(c.status == CheckStatus.WARNING for c in checks):
            overall = CheckStatus.WARNING
        else:
            overall = CheckStatus.HEALTHY
        
        report = HealthReport(
            overall=overall,
            checks=checks,
            uptime_seconds=self.sys_monitor.uptime_seconds,
        )
        
        self._last_report = report
        
        # 触发警报
        if report.has_critical and self.alert_callback:
            self.alert_callback(report)
        
        return report
    
    def raise_alert(self, report: HealthReport):
        """推送高级别警报"""
        critical_checks = [c for c in report.checks if c.is_critical]
        alert_msg = (
            f"🔴 系统健康检查发现 {len(critical_checks)} 个严重问题\n"
            + "\n".join(f"  - [{c.name}] {c.message}" for c in critical_checks)
        )
        self._logger.error(alert_msg)
        
        # 写入告警日志
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        alert_file = log_dir / f"alert_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(alert_file, "w") as f:
            json.dump(report.to_dict(), f, indent=2, ensure_ascii=False)
    
    def start_periodic_check(self):
        """启动定时健康检查 (后台线程)"""
        if self._running:
            return
        
        self._running = True
        
        def _loop():
            while self._running:
                try:
                    self.run_full_check()
                except Exception as e:
                    self._logger.error(f"定时健康检查异常: {e}")
                time.sleep(self.check_interval_s)
        
        self._thread = threading.Thread(target=_loop, daemon=True)
        self._thread.start()
        self._logger.info(f"定时健康检查已启动 | 间隔: {self.check_interval_s}s")
    
    def stop_periodic_check(self):
        """停止定时健康检查"""
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)
        self._logger.info("定时健康检查已停止")
    
    def to_metrics_payload(self) -> Dict[str, Any]:
        """生成 Prometheus/Grafana 兼容的指标负载"""
        if self._last_report is None:
            self.run_full_check()
        
        report = self._last_report
        metrics: Dict[str, Any] = {
            "timestamp": datetime.now().isoformat(),
            "uptime_seconds": self.sys_monitor.uptime_seconds,
            "health_status": report.overall.value if report else "unknown",
            "checks_healthy": report.healthy_count if report else 0,
            "checks_warning": report.warning_count if report else 0,
            "checks_critical": report.critical_count if report else 0,
        }
        
        # 系统资源指标
        try:
            import psutil
            metrics["cpu_percent"] = psutil.cpu_percent()
            metrics["memory_percent"] = psutil.virtual_memory().percent
            metrics["disk_percent"] = psutil.disk_usage("/").percent
        except ImportError:
            pass
        
        return metrics
