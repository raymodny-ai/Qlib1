"""
PM 熔断门控控制器 (PM Gate Controller)

对标 PRD 第6章:
- 投资组合经理 (PM) 拥有最高决策与一键熔断权
- 只有经过 PM 审批，调仓权重信号才允许推送到生产执行网关
- 门控状态与操作全程记录防篡改审计日志

核心组件:
- PMGateController: 门控核心，管理信号/训练/部署三路门控状态
- GateState: 门控状态枚举 (OPEN=放行, CLOSED=禁止)
- GateAction: 门控操作记录

状态机:
    OPEN ──(PM emergency_stop)──▶ CLOSED
    CLOSED ──(PM emergency_reopen)──▶ OPEN

门控维度:
    - signal_gate: 信号推送门控 (PM 一键熔断)
    - train_gate:  模型训练门控 (异常时自动/手动挂起)
    - deploy_gate: 模型部署门控 (仅 PM 可打开)

使用示例:
    from src.security.pm_gate import PMGateController, GateState

    gate = PMGateController(rbac=rbac, audit=audit)

    # PM 一键熔断
    gate.emergency_stop(user_id="pm_zhang", reason="市场剧烈波动，暂停信号推送")

    # 检查门控状态
    if gate.can_push_signal():
        push_to_production(signals)

    # PM 恢复
    gate.emergency_reopen(user_id="pm_zhang", reason="市场回归平稳，恢复信号推送")
"""

import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

from src.utils.logger import get_logger


# ============================================================================
#  门控状态
# ============================================================================

class GateState(str, Enum):
    """门控状态"""
    OPEN = "open"        # 放行
    CLOSED = "closed"    # 禁止/熔断


class GateDimension(str, Enum):
    """门控维度"""
    SIGNAL = "signal"    # 信号推送
    TRAIN = "train"      # 模型训练
    DEPLOY = "deploy"    # 模型部署


@dataclass
class GateAction:
    """门控操作记录"""
    action_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    dimension: str = ""              # signal | train | deploy
    action: str = ""                 # emergency_stop | emergency_reopen | auto_trip
    from_state: str = ""
    to_state: str = ""
    triggered_by: str = ""           # 操作者 user_id
    triggered_by_role: str = ""      # 操作者角色
    reason: str = ""                 # 原因
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "action_id": self.action_id,
            "dimension": self.dimension,
            "action": self.action,
            "from_state": self.from_state,
            "to_state": self.to_state,
            "triggered_by": self.triggered_by,
            "triggered_by_role": self.triggered_by_role,
            "reason": self.reason,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
        }


# ============================================================================
#  PMGateController — 熔断门控核心
# ============================================================================

class PMGateController:
    """
    PM 熔断门控控制器

    对标 PRD 第6章: PM 拥有一键熔断权，是信号推送的最终把关者。
    集成 RBAC 强制验证操作者身份 (仅 PM/SYSTEM_ADMIN 可操作门控)。

    三个独立门控维度:
    - signal_gate:  控制信号是否允许推送到生产执行网关
    - train_gate:   控制模型训练管线是否允许运行
    - deploy_gate:  控制模型是否允许部署到生产环境
    """

    def __init__(
        self,
        rbac=None,
        audit_logger=None,
        circuit_breaker=None,
        default_state: GateState = GateState.OPEN,
        auto_trip_on_alert: bool = True,
        min_action_interval: float = 5.0,
    ):
        """
        Args:
            rbac: RBACManager 实例 (用于身份验证)
            audit_logger: AuditLogger 实例 (用于操作审计)
            circuit_breaker: CircuitBreaker 实例 (可选，用于自动熔断联动)
            default_state: 初始门控状态
            auto_trip_on_alert: 是否在系统告警时自动熔断
            min_action_interval: 最小操作间隔 (秒)，防止短时间内反复操作
        """
        self._rbac = rbac
        self._audit_logger = audit_logger
        self._circuit_breaker = circuit_breaker
        self._auto_trip_on_alert = auto_trip_on_alert
        self._min_action_interval = min_action_interval

        # 三门控维度独立状态
        self._gates: Dict[str, GateState] = {
            GateDimension.SIGNAL: default_state,
            GateDimension.TRAIN: default_state,
            GateDimension.DEPLOY: GateState.CLOSED,  # 部署门默认关闭，需 PM 手动打开
        }

        # 操作历史
        self._history: List[GateAction] = []
        self._max_history: int = 500

        # 频率限制: 防止短时间内反复操作
        self._last_action_time: Dict[str, float] = {}

        self._lock = threading.RLock()
        self.logger = get_logger(__name__)

    # ------------------------------------------------------------------
    #  状态查询
    # ------------------------------------------------------------------

    @property
    def signal_gate(self) -> GateState:
        with self._lock:
            return self._gates[GateDimension.SIGNAL]

    @property
    def train_gate(self) -> GateState:
        with self._lock:
            return self._gates[GateDimension.TRAIN]

    @property
    def deploy_gate(self) -> GateState:
        with self._lock:
            return self._gates[GateDimension.DEPLOY]

    def get_gate_state(self, dimension: str) -> GateState:
        """获取指定维度门控状态"""
        with self._lock:
            return self._gates.get(dimension, GateState.OPEN)

    def get_all_states(self) -> Dict[str, str]:
        """获取三门控状态"""
        with self._lock:
            return {dim: state.value for dim, state in self._gates.items()}

    def can_push_signal(self) -> bool:
        """信号是否允许推送"""
        return self.signal_gate == GateState.OPEN

    def can_train_model(self) -> bool:
        """是否允许训练模型"""
        return self.train_gate == GateState.OPEN

    def can_deploy_model(self) -> bool:
        """是否允许部署模型"""
        return self.deploy_gate == GateState.OPEN

    def is_any_closed(self) -> bool:
        """是否有任意门控处于关闭 (熔断) 状态"""
        with self._lock:
            return any(s == GateState.CLOSED for s in self._gates.values())

    # ------------------------------------------------------------------
    #  RBAC 验证
    # ------------------------------------------------------------------

    def _verify_pm_authority(self, user_id: str) -> bool:
        """
        验证操作者是否有 PM 权限 (一键熔断权)

        PM 或 SYSTEM_ADMIN 可操作门控。
        """
        if self._rbac is None:
            # 无 RBAC 管理器时放行 (开发环境)
            return True

        from src.security.security import Role

        user = self._rbac.get_user(user_id) if hasattr(self._rbac, "get_user") else None
        if user is None:
            return False

        # 检查是否 PM 或 SYSTEM_ADMIN
        if hasattr(user, "role"):
            allowed_roles = {Role.PORTFOLIO_MANAGER, Role.SYSTEM_ADMIN}
            return user.role in allowed_roles

        # 备选: 通过 check_permission 验证
        if hasattr(self._rbac, "check_permission"):
            return (
                self._rbac.check_permission(user_id, "signal:emergency_stop")
                or self._rbac.check_permission(user_id, "signal:approve")
            )

        return False

    def _check_rate_limit(self, dimension: str) -> bool:
        """频率限制检查"""
        with self._lock:
            last_time = self._last_action_time.get(dimension, 0)
            elapsed = time.time() - last_time
            return elapsed >= self._min_action_interval

    def _record_rate_limit(self, dimension: str):
        """记录最近操作时间"""
        with self._lock:
            self._last_action_time[dimension] = time.time()

    # ------------------------------------------------------------------
    #  核心操作: 紧急熔断 (Emergency Stop)
    # ------------------------------------------------------------------

    def emergency_stop(
        self,
        user_id: str,
        dimension: str = GateDimension.SIGNAL,
        reason: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> GateAction:
        """
        PM 一键熔断 — 关闭指定维度的门控

        使用场景:
        - 市场剧烈波动，暂停信号推送
        - 发现模型预测严重偏差，挂起训练管线
        - 部署出现生产事故，立即阻断部署

        Args:
            user_id: 操作者 user_id
            dimension: 门控维度 (signal/train/deploy)
            reason: 熔断原因
            metadata: 附加元数据

        Returns:
            GateAction 操作记录

        Raises:
            PermissionError: 非 PM 无权操作
            RuntimeError: 频繁操作被限流
        """
        # 1. RBAC 验证
        if not self._verify_pm_authority(user_id):
            raise PermissionError(
                f"熔断操作被拒绝: user={user_id} 不是 Portfolio Manager，无权执行一键熔断"
            )

        # 2. 频率限制
        if not self._check_rate_limit(dimension):
            raise RuntimeError(
                f"操作过于频繁: dimension={dimension}，请等待 {self._min_action_interval} 秒后重试"
            )

        # 3. 查找用户角色
        role_str = "unknown"
        if self._rbac and hasattr(self._rbac, "get_user"):
            user = self._rbac.get_user(user_id)
            if user and hasattr(user, "role"):
                role_str = user.role.value if hasattr(user.role, "value") else str(user.role)

        # 4. 执行状态变更
        with self._lock:
            current_state = self._gates.get(dimension, GateState.OPEN)

            if current_state == GateState.CLOSED:
                raise RuntimeError(
                    f"门控 {dimension} 已处于熔断状态，无需重复操作"
                )

            self._gates[dimension] = GateState.CLOSED

            action = GateAction(
                dimension=dimension,
                action="emergency_stop",
                from_state=current_state.value,
                to_state=GateState.CLOSED.value,
                triggered_by=user_id,
                triggered_by_role=role_str,
                reason=reason,
                metadata=metadata or {},
            )
            self._history.append(action)
            if len(self._history) > self._max_history:
                self._history = self._history[-self._max_history:]

        self._record_rate_limit(dimension)

        # 5. 审计日志
        self._audit(
            event_type="pm_emergency_stop",
            user=user_id,
            role=role_str,
            action="emergency_stop",
            resource=f"gate/{dimension}",
            detail={
                "action_id": action.action_id,
                "dimension": dimension,
                "from_state": current_state.value,
                "to_state": GateState.CLOSED.value,
                "reason": reason,
            },
        )

        self.logger.warning(
            f"🚨 PM 一键熔断: dimension={dimension}, by={user_id}, reason={reason}"
        )

        # 6. 推送高级别告警 (可扩展)
        self._trigger_alert(
            level="critical",
            title=f"PM Emergency Stop: {dimension}",
            message=f"User '{user_id}' triggered emergency stop on {dimension}: {reason}",
        )

        return action

    # ------------------------------------------------------------------
    #  核心操作: 恢复放行 (Emergency Reopen)
    # ------------------------------------------------------------------

    def emergency_reopen(
        self,
        user_id: str,
        dimension: str = GateDimension.SIGNAL,
        reason: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> GateAction:
        """
        PM 恢复放行 — 重新打开指定维度的门控

        使用场景:
        - 市场回归平稳，恢复信号推送
        - 模型问题修复完毕，恢复训练管线
        - 生产事故解决，恢复部署

        Args:
            user_id: 操作者 user_id
            dimension: 门控维度 (signal/train/deploy)
            reason: 恢复原因
            metadata: 附加元数据

        Returns:
            GateAction 操作记录

        Raises:
            PermissionError: 非 PM 无权操作
        """
        # 1. RBAC 验证
        if not self._verify_pm_authority(user_id):
            raise PermissionError(
                f"恢复操作被拒绝: user={user_id} 不是 Portfolio Manager，无权恢复放行"
            )

        # 3. 查找用户角色
        role_str = "unknown"
        if self._rbac and hasattr(self._rbac, "get_user"):
            user = self._rbac.get_user(user_id)
            if user and hasattr(user, "role"):
                role_str = user.role.value if hasattr(user.role, "value") else str(user.role)

        # 4. 执行状态变更
        with self._lock:
            current_state = self._gates.get(dimension, GateState.CLOSED)

            if current_state == GateState.OPEN:
                raise RuntimeError(
                    f"门控 {dimension} 已处于放行状态，无需重复操作"
                )

            self._gates[dimension] = GateState.OPEN

            action = GateAction(
                dimension=dimension,
                action="emergency_reopen",
                from_state=current_state.value,
                to_state=GateState.OPEN.value,
                triggered_by=user_id,
                triggered_by_role=role_str,
                reason=reason,
                metadata=metadata or {},
            )
            self._history.append(action)
            if len(self._history) > self._max_history:
                self._history = self._history[-self._max_history:]

        self._record_rate_limit(dimension)

        # 5. 审计日志
        self._audit(
            event_type="pm_emergency_reopen",
            user=user_id,
            role=role_str,
            action="emergency_reopen",
            resource=f"gate/{dimension}",
            detail={
                "action_id": action.action_id,
                "dimension": dimension,
                "from_state": current_state.value,
                "to_state": GateState.OPEN.value,
                "reason": reason,
            },
        )

        self.logger.info(
            f"✅ PM 恢复放行: dimension={dimension}, by={user_id}, reason={reason}"
        )

        # 6. 推送恢复通知
        self._trigger_alert(
            level="info",
            title=f"PM Reopened Gate: {dimension}",
            message=f"User '{user_id}' reopened {dimension} gate: {reason}",
        )

        return action

    # ------------------------------------------------------------------
    #  便捷操作: 全局紧急熔断 / 全局恢复
    # ------------------------------------------------------------------

    def global_emergency_stop(
        self,
        user_id: str,
        reason: str = "",
    ) -> List[GateAction]:
        """
        PM 全局紧急熔断 — 同时关闭所有三门控

        最严重场景: 系统性风险、交易所停摆等。
        """
        actions = []
        for dim in [GateDimension.SIGNAL, GateDimension.TRAIN, GateDimension.DEPLOY]:
            try:
                if self._gates[dim] == GateState.CLOSED:
                    continue
                action = self.emergency_stop(
                    user_id=user_id,
                    dimension=dim,
                    reason=f"[GLOBAL STOP] {reason}",
                )
                actions.append(action)
            except Exception as e:
                self.logger.error(f"全局熔断失败: dim={dim}, error={e}")

        self.logger.critical(
            f"🚨🚨🚨 PM 全局紧急熔断: by={user_id}, reason={reason}, "
            f"affected_dimensions={len(actions)}"
        )

        return actions

    def global_emergency_reopen(
        self,
        user_id: str,
        reason: str = "",
    ) -> List[GateAction]:
        """PM 全局恢复"""
        actions = []
        for dim in [GateDimension.SIGNAL, GateDimension.TRAIN, GateDimension.DEPLOY]:
            try:
                if self._gates[dim] == GateState.OPEN:
                    continue
                action = self.emergency_reopen(
                    user_id=user_id,
                    dimension=dim,
                    reason=f"[GLOBAL REOPEN] {reason}",
                )
                actions.append(action)
            except Exception as e:
                self.logger.error(f"全局恢复失败: dim={dim}, error={e}")

        self.logger.info(
            f"✅✅✅ PM 全局恢复放行: by={user_id}, reason={reason}"
        )

        return actions

    # ------------------------------------------------------------------
    #  自动熔断 (Auto-Trip)
    # ------------------------------------------------------------------

    def auto_trip(
        self,
        dimension: str = GateDimension.SIGNAL,
        reason: str = "",
        source: str = "system",
    ) -> Optional[GateAction]:
        """
        系统自动熔断 (无需 PM 权限)

        由系统内部异常触发:
        - CircuitBreaker 开路
        - 健康检查连续失败
        - PIT 数据完整性校验失败
        - 因子断层检测告警

        Args:
            dimension: 门控维度
            reason: 自动熔断原因
            source: 触发源 (如 circuit_breaker, health_check, pit_validator)

        Returns:
            GateAction 或 None (已熔断时)
        """
        if not self._auto_trip_on_alert:
            self.logger.debug(f"自动熔断已禁用，忽略: {dimension}, reason={reason}")
            return None

        with self._lock:
            current_state = self._gates.get(dimension, GateState.OPEN)
            if current_state == GateState.CLOSED:
                return None

            self._gates[dimension] = GateState.CLOSED

            action = GateAction(
                dimension=dimension,
                action="auto_trip",
                from_state=current_state.value,
                to_state=GateState.CLOSED.value,
                triggered_by="system",
                triggered_by_role="system",
                reason=reason,
                metadata={"source": source},
            )
            self._history.append(action)

        self._audit(
            event_type="gate_auto_trip",
            user="system",
            role="system",
            action="auto_trip",
            resource=f"gate/{dimension}",
            detail={
                "action_id": action.action_id,
                "dimension": dimension,
                "reason": reason,
                "source": source,
            },
        )

        self.logger.warning(
            f"⚡ 自动熔断: dimension={dimension}, source={source}, reason={reason}"
        )

        self._trigger_alert(
            level="warning",
            title=f"Auto-Trip: {dimension}",
            message=f"System auto-tripped {dimension} gate: {reason} (source: {source})",
        )

        return action

    # ------------------------------------------------------------------
    #  历史查询
    # ------------------------------------------------------------------

    def get_history(
        self,
        dimension: Optional[str] = None,
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        """查询门控操作历史"""
        with self._lock:
            entries = self._history
            if dimension:
                entries = [e for e in entries if e.dimension == dimension]
            return [e.to_dict() for e in entries[-limit:]]

    def get_recent_stops(self, hours: int = 24) -> List[Dict[str, Any]]:
        """获取近期熔断记录"""
        cutoff = datetime.now(timezone.utc).isoformat()
        with self._lock:
            return [
                e.to_dict() for e in self._history
                if e.action in ("emergency_stop", "auto_trip")
                and e.timestamp >= cutoff
            ]

    def get_stats(self) -> Dict[str, Any]:
        """获取门控统计"""
        with self._lock:
            stops = [e for e in self._history if e.action in ("emergency_stop", "auto_trip")]
            reopens = [e for e in self._history if e.action == "emergency_reopen"]
            auto_trips = [e for e in self._history if e.action == "auto_trip"]

            return {
                "current_states": self.get_all_states(),
                "total_actions": len(self._history),
                "total_stops": len(stops),
                "total_reopens": len(reopens),
                "total_auto_trips": len(auto_trips),
                "is_any_closed": self.is_any_closed(),
                "recent_24h_stops": len(self.get_recent_stops(hours=24)),
            }

    # ------------------------------------------------------------------
    #  装饰器: 门控保护
    # ------------------------------------------------------------------

    def require_signal_gate_open(self, func: Callable) -> Callable:
        """
        装饰器: 要求信号门控打开才能执行

        用法:
            @gate.require_signal_gate_open
            def push_signals(signals):
                ...
        """
        import functools

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if not self.can_push_signal():
                raise GateClosedError(
                    f"信号推送被熔断: signal_gate is {self.signal_gate.value}. "
                    f"请联系 Portfolio Manager 恢复放行。"
                )
            return func(*args, **kwargs)
        return wrapper

    def require_train_gate_open(self, func: Callable) -> Callable:
        """装饰器: 要求训练门控打开"""
        import functools

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if not self.can_train_model():
                raise GateClosedError(
                    f"模型训练被挂起: train_gate is {self.train_gate.value}"
                )
            return func(*args, **kwargs)
        return wrapper

    def require_deploy_gate_open(self, func: Callable) -> Callable:
        """装饰器: 要求部署门控打开"""
        import functools

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if not self.can_deploy_model():
                raise GateClosedError(
                    f"模型部署被阻断: deploy_gate is {self.deploy_gate.value}"
                )
            return func(*args, **kwargs)
        return wrapper

    # ------------------------------------------------------------------
    #  告警与重置
    # ------------------------------------------------------------------

    def _trigger_alert(self, level: str, title: str, message: str):
        """推送告警 (可对接外部告警系统如 PagerDuty/Slack)"""
        alert = {
            "level": level,
            "title": title,
            "message": message,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "component": "pm_gate_controller",
        }
        # 写入告警日志
        try:
            import json
            from pathlib import Path
            alert_dir = Path("logs")
            alert_dir.mkdir(parents=True, exist_ok=True)
            alert_file = alert_dir / f"alert_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(alert_file, "w") as f:
                json.dump(alert, f, indent=2, ensure_ascii=False)
        except Exception:
            pass

    def _audit(
        self,
        event_type: str,
        user: str,
        role: str = "",
        action: str = "",
        resource: str = "",
        detail: Optional[Dict[str, Any]] = None,
    ):
        """写入审计日志"""
        if self._audit_logger and hasattr(self._audit_logger, "log"):
            self._audit_logger.log(
                event_type=event_type,
                user=user,
                role=role,
                action=action,
                resource=resource,
                detail=detail or {},
            )

    def reset(self):
        """重置所有门控到默认状态 (仅开发/测试用)"""
        with self._lock:
            self._gates[GateDimension.SIGNAL] = GateState.OPEN
            self._gates[GateDimension.TRAIN] = GateState.OPEN
            self._gates[GateDimension.DEPLOY] = GateState.CLOSED
            self._history.clear()
            self._last_action_time.clear()
        self.logger.warning("门控已重置到默认状态 (仅限开发/测试)")


class GateClosedError(Exception):
    """门控关闭异常 — 操作被熔断门控拦截"""
    pass
