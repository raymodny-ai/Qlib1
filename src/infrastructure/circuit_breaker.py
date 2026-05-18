"""
熔断器与自愈机制 (Circuit Breaker & Auto-Healing)

对标 PRD 4.3 全天候系统稳定性与故障自愈能力：
- 99.9% 高可用性 SLA
- DataHandler 异常值免疫与熔断隔离
- 自动化健康探针 (每日落盘后核查 PIT 时序单调性与因子断层)
- 严重跳空时立即挂起模型重训管线并推送高级别警报

核心组件:
- CircuitBreaker: 熔断器 (Closed → Open → HalfOpen)
- RetryPolicy: 重试策略 (指数退避/抖动)
- SLA Tracker: 99.9% 可用性 SLA 追踪器
- AutoHealingManager: 自动恢复编排器

使用示例:
    from src.infrastructure.circuit_breaker import CircuitBreaker, RetryPolicy
    
    breaker = CircuitBreaker(name="data_handler", failure_threshold=5)
    retry = RetryPolicy(max_retries=3, backoff_base=2.0)
    
    @breaker
    @retry
    def load_data(instrument, field):
        return ds.load_field(instrument, field)
"""

import functools
import random
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

from src.utils.logger import get_logger


# ============================================================================
#  枚举与常量
# ============================================================================

class CircuitState(str, Enum):
    """熔断器状态"""
    CLOSED = "closed"          # 正常通行
    OPEN = "open"              # 熔断开路，拒绝请求
    HALF_OPEN = "half_open"    # 半开，试探性放行


@dataclass
class BreakerEvent:
    """熔断器事件"""
    timestamp: float = field(default_factory=time.time)
    state: CircuitState = CircuitState.CLOSED
    event: str = ""
    detail: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SlaRecord:
    """SLA 记录"""
    timestamp: float = field(default_factory=time.time)
    service: str = ""
    status: str = "up"   # "up" | "degraded" | "down"
    response_time_ms: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


# ============================================================================
#  CircuitBreaker — 熔断器
# ============================================================================

class CircuitBreaker:
    """
    熔断器 (Circuit Breaker Pattern)

    状态机:
      CLOSED ──(failures > threshold)──▶ OPEN
      OPEN   ──(cooldown elapsed)──────▶ HALF_OPEN
      HALF_OPEN ──(success)────────────▶ CLOSED
      HALF_OPEN ──(failure)────────────▶ OPEN

    使用示例:
        breaker = CircuitBreaker(name="data_fetch", failure_threshold=3)

        @breaker.protect
        def fetch_data(url):
            ...

        # 或装饰器模式
        @breaker
        def fetch_data(url):
            ...

        # 上下文管理器
        with breaker:
            result = risky_operation()
    """

    def __init__(
        self,
        name: str = "default",
        failure_threshold: int = 5,
        cooldown_seconds: float = 30.0,
        half_open_max_requests: int = 3,
        exception_types: Tuple[type, ...] = (Exception,),
        on_open: Optional[Callable[[str], None]] = None,
        on_close: Optional[Callable[[str], None]] = None,
    ):
        """
        Args:
            name: 熔断器名称
            failure_threshold: 连续失败阈值 (达到后断开)
            cooldown_seconds: 断开后的冷却时间
            half_open_max_requests: 半开状态下最大试探请求数
            exception_types: 计入失败的异常类型
            on_open: 熔断开路回调
            on_close: 熔断闭合回调
        """
        self.name = name
        self.failure_threshold = failure_threshold
        self.cooldown_seconds = cooldown_seconds
        self.half_open_max_requests = half_open_max_requests
        self.exception_types = exception_types
        self.on_open = on_open
        self.on_close = on_close

        self._state: CircuitState = CircuitState.CLOSED
        self._failure_count: int = 0
        self._success_count: int = 0
        self._total_requests: int = 0
        self._total_failures: int = 0
        self._total_successes: int = 0
        self._last_failure_time: Optional[float] = None
        self._last_success_time: Optional[float] = None
        self._opened_at: Optional[float] = None
        self._half_open_requests: int = 0
        self._lock = threading.RLock()
        self._events: List[BreakerEvent] = []
        self._max_events: int = 100
        self.logger = get_logger()

    # ------------------------------------------------------------------
    #  状态查询
    # ------------------------------------------------------------------

    @property
    def state(self) -> CircuitState:
        with self._lock:
            return self._state

    @property
    def is_open(self) -> bool:
        return self.state == CircuitState.OPEN

    @property
    def failure_rate(self) -> float:
        total = self._total_failures + self._total_successes
        return self._total_failures / total if total > 0 else 0.0

    @property
    def health_score(self) -> float:
        """健康分数 0-100"""
        with self._lock:
            if self._state == CircuitState.OPEN:
                return 0.0
            if self._total_requests == 0:
                return 100.0
            success_rate = self._total_successes / self._total_requests
            return round(success_rate * 100.0, 1)

    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        with self._lock:
            return {
                "name": self.name,
                "state": self._state.value,
                "failure_count": self._failure_count,
                "total_requests": self._total_requests,
                "total_failures": self._total_failures,
                "total_successes": self._total_successes,
                "failure_rate": round(self.failure_rate, 4),
                "health_score": self.health_score,
                "last_failure_time": self._last_failure_time,
                "last_success_time": self._last_success_time,
            }

    # ------------------------------------------------------------------
    #  核心状态机
    # ------------------------------------------------------------------

    def _before_request(self) -> bool:
        """
        请求前检查，返回是否允许通过

        Returns:
            True = 允许通过, False = 拒绝
        """
        with self._lock:
            if self._state == CircuitState.CLOSED:
                return True

            if self._state == CircuitState.OPEN:
                elapsed = time.time() - (self._opened_at or 0)
                if elapsed >= self.cooldown_seconds:
                    self._transition_to(CircuitState.HALF_OPEN)
                    self._half_open_requests = 0
                    return True
                return False

            if self._state == CircuitState.HALF_OPEN:
                if self._half_open_requests < self.half_open_max_requests:
                    self._half_open_requests += 1
                    return True
                return False

            return False

    def _on_success(self):
        """请求成功后回调"""
        with self._lock:
            self._total_requests += 1
            self._total_successes += 1
            self._success_count += 1
            self._failure_count = 0
            self._last_success_time = time.time()

            if self._state == CircuitState.HALF_OPEN:
                self._transition_to(CircuitState.CLOSED)

    def _on_failure(self, exception: Exception):
        """请求失败后回调"""
        with self._lock:
            self._total_requests += 1
            self._total_failures += 1
            self._failure_count += 1
            self._last_failure_time = time.time()

            if self._state == CircuitState.CLOSED and self._failure_count >= self.failure_threshold:
                self._transition_to(CircuitState.OPEN)

            if self._state == CircuitState.HALF_OPEN:
                self._transition_to(CircuitState.OPEN)

    def _transition_to(self, new_state: CircuitState):
        """状态转换"""
        old_state = self._state
        self._state = new_state

        event = BreakerEvent(state=new_state, event=f"{old_state.value} → {new_state.value}")
        self._events.append(event)
        if len(self._events) > self._max_events:
            self._events = self._events[-self._max_events:]

        if new_state == CircuitState.OPEN:
            self._opened_at = time.time()
            self._half_open_requests = 0
            self.logger.warning("熔断器开路", name=self.name, failures=self._failure_count)

        if new_state == CircuitState.CLOSED:
            self._failure_count = 0
            self.logger.info("熔断器闭合", name=self.name)

        # 回调
        if new_state == CircuitState.OPEN and self.on_open:
            try:
                self.on_open(self.name)
            except Exception:
                pass
        if new_state == CircuitState.CLOSED and self.on_close:
            try:
                self.on_close(self.name)
            except Exception:
                pass

    # ------------------------------------------------------------------
    #  快捷接口
    # ------------------------------------------------------------------

    def __call__(self, func: Callable) -> Callable:
        """装饰器模式"""
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return self.call(func, *args, **kwargs)
        return wrapper

    protect = property(lambda self: self.__call__)

    def __enter__(self):
        if not self._before_request():
            raise CircuitBreakerOpenError(
                f"熔断器 [{self.name}] 处于 {self._state.value} 状态，请求被拒绝"
            )
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is None:
            self._on_success()
        elif exc_type and issubclass(exc_type, self.exception_types):
            self._on_failure(exc_val)
        elif exc_type:
            # 不计入熔断统计的异常，向上传播
            return False
        return False

    def call(self, func: Callable, *args, **kwargs) -> Any:
        """显式调用 (通过熔断器执行函数)"""
        with self:
            return func(*args, **kwargs)

    def reset(self):
        """手动重置熔断器"""
        with self._lock:
            self._state = CircuitState.CLOSED
            self._failure_count = 0
            self._half_open_requests = 0
            self._opened_at = None


class CircuitBreakerOpenError(Exception):
    """熔断器开路异常"""
    pass


# ============================================================================
#  RetryPolicy — 重试策略
# ============================================================================

class RetryPolicy:
    """
    指数退避重试策略

    支持:
    - 指数退避 (exponential backoff)
    - 随机抖动 (jitter)
    - 最大重试次数限制
    - 可重试异常类型过滤

    使用示例:
        retry = RetryPolicy(max_retries=3, backoff_base=2.0, jitter=True)

        # 装饰器
        @retry
        def unreliable_api(url):
            ...

        # 显式调用
        result = retry.call(unreliable_api, "https://api.example.com")
    """

    def __init__(
        self,
        max_retries: int = 3,
        backoff_base: float = 2.0,
        initial_delay: float = 1.0,
        max_delay: float = 60.0,
        jitter: bool = True,
        exception_types: Tuple[type, ...] = (Exception,),
        on_retry: Optional[Callable[[Exception, int, float], None]] = None,
    ):
        """
        Args:
            max_retries: 最大重试次数
            backoff_base: 退避基数 (2.0 = 指数退避)
            initial_delay: 初始延迟 (秒)
            max_delay: 最大延迟上限 (秒)
            jitter: 是否启用随机抖动
            exception_types: 可重试的异常类型
            on_retry: 重试回调 callback(exception, attempt, delay)
        """
        self.max_retries = max_retries
        self.backoff_base = backoff_base
        self.initial_delay = initial_delay
        self.max_delay = max_delay
        self.jitter = jitter
        self.exception_types = exception_types
        self.on_retry = on_retry

        self._total_retries: int = 0
        self._total_attempts: int = 0
        self._lock = threading.Lock()
        self.logger = get_logger()

    def compute_delay(self, attempt: int) -> float:
        """
        计算第 N 次重试的延迟

        delay = min(initial_delay * backoff_base^(attempt - 1), max_delay)
        """
        delay = self.initial_delay * (self.backoff_base ** (attempt - 1))
        delay = min(delay, self.max_delay)

        if self.jitter:
            # ±25% 随机抖动
            jitter_range = delay * 0.25
            delay += random.uniform(-jitter_range, jitter_range)
            delay = max(0, delay)

        return delay

    def should_retry(self, exception: Exception) -> bool:
        """判断异常是否可重试"""
        return isinstance(exception, self.exception_types)

    def __call__(self, func: Callable) -> Callable:
        """装饰器模式"""
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return self.call(func, *args, **kwargs)
        return wrapper

    def call(self, func: Callable, *args, **kwargs) -> Any:
        """
        执行函数，失败时自动重试

        Raises:
            最后一次失败的异常 (重试耗尽后)
        """
        last_exception = None

        for attempt in range(1, self.max_retries + 2):  # 1 initial + N retries
            try:
                with self._lock:
                    self._total_attempts += 1
                return func(*args, **kwargs)
            except self.exception_types as e:
                last_exception = e
                if attempt <= self.max_retries:
                    delay = self.compute_delay(attempt)
                    with self._lock:
                        self._total_retries += 1
                    self.logger.debug(
                        "重试",
                        attempt=attempt,
                        max_retries=self.max_retries,
                        delay=round(delay, 3),
                        error=str(e),
                    )
                    if self.on_retry:
                        try:
                            self.on_retry(e, attempt, delay)
                        except Exception:
                            pass
                    time.sleep(delay)
                else:
                    self.logger.error(
                        "重试耗尽",
                        attempts=attempt,
                        error=str(e),
                    )
                    raise

        raise last_exception  # type: ignore

    def get_stats(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "total_attempts": self._total_attempts,
                "total_retries": self._total_retries,
                "max_retries": self.max_retries,
                "retry_rate": round(
                    self._total_retries / self._total_attempts
                    if self._total_attempts > 0 else 0, 4
                ),
            }


# ============================================================================
#  SLA Tracker — 可用性 SLA 追踪器
# ============================================================================

class SlaTracker:
    """
    99.9% 可用性 SLA 追踪器

    对标 PRD 4.3: 系统核心微服务需达成 99.9% 高可用性。
    99.9% up = 每月最多 43.2 分钟宕机 (30天) 或 ~8.76 小时/年。

    使用示例:
        tracker = SlaTracker(target_sla=0.999)
        tracker.record_up("data_server", response_time_ms=12.5)
        tracker.record_down("data_server", reason="connection_timeout")
        print(f"Current SLA: {tracker.get_sla('data_server'):.4%}")
    """

    _target_sla: float = 0.999  # 99.9%

    def __init__(self, target_sla: float = 0.999, window_days: int = 30):
        """
        Args:
            target_sla: 目标可用性 (0.999 = 99.9%)
            window_days: 统计窗口 (天)
        """
        self.target_sla = target_sla
        self.window_days = window_days
        self._records: Dict[str, List[SlaRecord]] = {}
        self._lock = threading.RLock()
        self.logger = get_logger()

    def record_up(
        self,
        service: str,
        response_time_ms: float = 0.0,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """记录服务正常"""
        record = SlaRecord(
            service=service,
            status="up",
            response_time_ms=response_time_ms,
            metadata=metadata or {},
        )
        with self._lock:
            self._records.setdefault(service, []).append(record)
            self._prune(service)

    def record_down(
        self,
        service: str,
        reason: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """记录服务宕机"""
        meta = metadata or {}
        meta["reason"] = reason
        record = SlaRecord(
            service=service,
            status="down",
            metadata=meta,
        )
        with self._lock:
            self._records.setdefault(service, []).append(record)
            self._prune(service)
        self.logger.warning("SLA 宕机记录", service=service, reason=reason)

    def record_degraded(
        self,
        service: str,
        response_time_ms: float = 0.0,
        reason: str = "",
    ):
        """记录服务降级"""
        record = SlaRecord(
            service=service,
            status="degraded",
            response_time_ms=response_time_ms,
            metadata={"reason": reason},
        )
        with self._lock:
            self._records.setdefault(service, []).append(record)
            self._prune(service)

    def get_sla(self, service: str) -> float:
        """
        获取服务的当前 SLA (可用率)

        Returns:
            0-1 之间的浮点数, 如 0.999 = 99.9%
        """
        with self._lock:
            records = self._records.get(service, [])
            self._prune(service)
            records = self._records.get(service, [])

            total = len(records)
            up_count = sum(1 for r in records if r.status == "up")

            # 降级算 50% 可用
            degraded_count = sum(1 for r in records if r.status == "degraded")
            effective_up = up_count + degraded_count * 0.5

            return effective_up / total if total > 0 else 1.0

    def is_sla_breached(self, service: str) -> bool:
        """检查 SLA 是否跌破目标"""
        return self.get_sla(service) < self.target_sla

    def is_99_9_available(self, service: str) -> bool:
        """检查是否达到 99.9% 可用性"""
        return self.get_sla(service) >= 0.999

    def get_sla_report(self) -> Dict[str, Any]:
        """生成 SLA 报告"""
        with self._lock:
            report = {
                "target_sla": self.target_sla,
                "window_days": self.window_days,
                "generated_at": datetime.now().isoformat(),
                "services": {},
            }
            all_meet = True
            for service in self._records:
                self._prune(service)
                records = self._records.get(service, [])
                if not records:
                    continue
                up_count = sum(1 for r in records if r.status == "up")
                degraded_count = sum(1 for r in records if r.status == "degraded")
                down_count = sum(1 for r in records if r.status == "down")
                total = len(records)
                sla = self.get_sla(service)
                meets = sla >= self.target_sla
                if not meets:
                    all_meet = False

                # 平均响应时间
                up_times = [r.response_time_ms for r in records if r.status == "up" and r.response_time_ms > 0]
                avg_response_ms = sum(up_times) / len(up_times) if up_times else 0

                report["services"][service] = {
                    "sla": round(sla, 6),
                    "sla_pct": f"{sla*100:.3f}%",
                    "meets_target": meets,
                    "total_checks": total,
                    "up": up_count,
                    "degraded": degraded_count,
                    "down": down_count,
                    "avg_response_ms": round(avg_response_ms, 2),
                }

            report["all_meet_target"] = all_meet
            return report

    def _prune(self, service: str):
        """清理超出窗口的旧记录"""
        if service not in self._records:
            return
        cutoff = time.time() - self.window_days * 86400
        self._records[service] = [
            r for r in self._records[service]
            if r.timestamp >= cutoff
        ]

    def reset(self, service: Optional[str] = None):
        """重置统计数据"""
        with self._lock:
            if service:
                self._records.pop(service, None)
            else:
                self._records.clear()


# ============================================================================
#  AutoHealingManager — 自动恢复编排器
# ============================================================================

@dataclass
class HealingAction:
    """自愈动作"""
    name: str
    action: Callable[[], Any]
    priority: int = 0  # 数字越小越优先
    is_reversible: bool = True


@dataclass
class HealingResult:
    """自愈结果"""
    action_name: str
    success: bool
    error: Optional[str] = None
    duration_ms: float = 0.0


class AutoHealingManager:
    """
    自动恢复编排器

    当检测到系统异常时，按优先级尝试自愈动作：
    1. 缓存清理 → 2. 连接重置 → 3. 服务重启 → 4. 降级运行

    使用示例:
        manager = AutoHealingManager()

        manager.register_action(HealingAction(
            name="clear_cache",
            action=lambda: ds.clear_cache(),
            priority=1,
        ))
        manager.register_action(HealingAction(
            name="reconnect",
            action=lambda: ds.reconnect(),
            priority=2,
        ))

        results = manager.heal()
        if manager.is_healthy:
            log.info("System healed!")
    """

    def __init__(self, check_health: Optional[Callable[[], bool]] = None):
        """
        Args:
            check_health: 健康检查函数，返回 True = 健康
        """
        self._actions: Dict[str, HealingAction] = {}
        self._healing_history: List[HealingResult] = []
        self._max_history: int = 100
        self._lock = threading.Lock()
        self.check_health = check_health
        self.logger = get_logger()

    def register_action(self, action: HealingAction):
        """注册自愈动作"""
        with self._lock:
            self._actions[action.name] = action

    def unregister_action(self, name: str):
        """移除自愈动作"""
        with self._lock:
            self._actions.pop(name, None)

    @property
    def is_healthy(self) -> bool:
        """检查系统是否健康"""
        if self.check_health:
            try:
                return self.check_health()
            except Exception:
                return False
        return True

    def heal(self, max_rounds: int = 3) -> List[HealingResult]:
        """
        执行自愈流程

        按优先级逐动作尝试，每轮后检查健康状态。
        最多执行 max_rounds 轮。

        Returns:
            自愈结果列表
        """
        results: List[HealingResult] = []

        # 健康检查通过则无需自愈
        if self.is_healthy:
            self.logger.info("系统健康，无需自愈")
            return results

        self.logger.warning("启动自愈流程", max_rounds=max_rounds)

        # 按优先级排序
        with self._lock:
            sorted_actions = sorted(
                self._actions.values(),
                key=lambda a: a.priority,
            )

        for round_num in range(max_rounds):
            for action in sorted_actions:
                t0 = time.perf_counter()
                try:
                    action.action()
                    success = True
                    error = None
                    self.logger.info("自愈动作成功", action=action.name)
                except Exception as e:
                    success = False
                    error = str(e)
                    self.logger.error("自愈动作失败", action=action.name, error=error)

                result = HealingResult(
                    action_name=action.name,
                    success=success,
                    error=error,
                    duration_ms=(time.perf_counter() - t0) * 1000,
                )
                results.append(result)

                with self._lock:
                    self._healing_history.append(result)
                    if len(self._healing_history) > self._max_history:
                        self._healing_history = self._healing_history[-self._max_history:]

                # 健康检查
                if self.is_healthy:
                    self.logger.info("自愈成功", actions_executed=len(results))
                    return results

            # 短暂延迟后进行下一轮
            if round_num < max_rounds - 1:
                time.sleep(2.0)

        self.logger.error("自愈失败，已达最大轮次", rounds=max_rounds)
        return results

    def get_history(self) -> List[Dict[str, Any]]:
        with self._lock:
            return [
                {"action": r.action_name, "success": r.success,
                 "error": r.error, "duration_ms": round(r.duration_ms, 2)}
                for r in self._healing_history
            ]

    def get_stats(self) -> Dict[str, Any]:
        with self._lock:
            total = len(self._healing_history)
            successes = sum(1 for r in self._healing_history if r.success)
            return {
                "total_healings": total,
                "successes": successes,
                "failures": total - successes,
                "success_rate": round(successes / total, 4) if total > 0 else 1.0,
            }


# ============================================================================
#  熔断自愈装饰器 (CircuitBreaker + RetryPolicy)
# ============================================================================

def with_resilience(
    breaker: Optional[CircuitBreaker] = None,
    retry: Optional[RetryPolicy] = None,
) -> Callable:
    """
    组合断路器与重试策略的装饰器工厂

    执行顺序: Retry → CircuitBreaker → Function
    """
    def decorator(func: Callable) -> Callable:
        wrapped = func
        if retry is not None:
            wrapped = retry(wrapped)
        if breaker is not None:
            wrapped = breaker(wrapped)
        return functools.wraps(func)(wrapped)
    return decorator
