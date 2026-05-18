"""
信号导出与生产层接口 (Signal Exporter & Production Interface)

实现研究层到生产执行层的安全信号推送。将模型预测的
目标股票池与资金配置权重，通过加密通道推送至物理隔离的
生产执行层 (如 QuantConnect LEAN 或 OMS)。

核心组件:
- SignalExporter: 信号序列化、加密与推送
- SignalFormat: 标准化信号格式
- OMSAdapter: 订单管理系统适配器
- ProductionGateway: 生产环境安全网关

设计原则:
- 研究层与生产层网络隔离
- 全链路 TLS 1.2+ 加密传输
- 信号防篡改签名
- 支持多种目标系统 (LEAN/OMS/自定义)

使用示例:
    from src.infrastructure.signal_exporter import SignalExporter
    
    exporter = SignalExporter(encryptor=aes, target_url="https://oms.internal/api/signals")
    exporter.push(predictions, weights, approved_by="pm_zhang")
"""

import hashlib
import hmac
import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np
import pandas as pd

from src.utils.logger import get_logger


# ============================================================================
#  信号格式定义
# ============================================================================

@dataclass
class SignalEntry:
    """单条交易信号"""
    instrument: str            # 股票代码
    action: str                # BUY | SELL | HOLD
    target_weight: float       # 目标权重 [0, 1]
    score: float               # 预测得分
    quantity: int = 0          # 建议数量
    limit_price: Optional[float] = None  # 限价 (可选)
    order_type: str = "MKT"    # MKT | LMT


@dataclass
class SignalBatch:
    """
    标准化信号批次
    
    每天开盘前由研究层生成，推送至生产执行层。
    """
    batch_id: str = field(default_factory=lambda: f"sig_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}")
    generated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    model_name: str = ""
    model_version: str = ""
    approved_by: str = ""              # 审批人 (PM)
    approved_at: str = ""
    signals: List[SignalEntry] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # 安全字段
    signature: str = ""                # HMAC 签名
    encryption_algorithm: str = "AES-256-GCM"
    
    def to_payload(self) -> Dict[str, Any]:
        """转为可序列化的传输负载"""
        return {
            "batch_id": self.batch_id,
            "generated_at": self.generated_at,
            "model_name": self.model_name,
            "model_version": self.model_version,
            "approved_by": self.approved_by,
            "approved_at": self.approved_at,
            "signals": [
                {
                    "instrument": s.instrument,
                    "action": s.action,
                    "target_weight": round(s.target_weight, 6),
                    "score": round(s.score, 6),
                    "quantity": s.quantity,
                    "limit_price": round(s.limit_price, 6) if s.limit_price else None,
                    "order_type": s.order_type,
                }
                for s in self.signals
            ],
            "metadata": self.metadata,
            "signature": self.signature,
            "encryption_algorithm": self.encryption_algorithm,
        }
    
    @classmethod
    def from_payload(cls, payload: Dict[str, Any]) -> "SignalBatch":
        """从传输负载还原"""
        signals = [
            SignalEntry(
                instrument=s["instrument"],
                action=s["action"],
                target_weight=s["target_weight"],
                score=s["score"],
                quantity=s.get("quantity", 0),
                limit_price=s.get("limit_price"),
                order_type=s.get("order_type", "MKT"),
            )
            for s in payload.get("signals", [])
        ]
        
        return cls(
            batch_id=payload["batch_id"],
            generated_at=payload["generated_at"],
            model_name=payload.get("model_name", ""),
            model_version=payload.get("model_version", ""),
            approved_by=payload.get("approved_by", ""),
            approved_at=payload.get("approved_at", ""),
            signals=signals,
            metadata=payload.get("metadata", {}),
            signature=payload.get("signature", ""),
        )
    
    def validate(self) -> Tuple[bool, str]:
        """
        验证信号批次的合法性
        
        Returns:
            (is_valid, message)
        """
        if not self.signals:
            return False, "信号列表为空"
        
        if not self.approved_by:
            return False, "信号未经 PM 审批"
        
        # 检查权重总和 ≤ 1
        total_weight = sum(s.target_weight for s in self.signals)
        if total_weight > 1.01:  # 允许 1% 浮点容差
            return False, f"权重总和 {total_weight:.4f} 超过 1.0"
        
        # 检查是否有重复标的
        instruments = [s.instrument for s in self.signals]
        if len(instruments) != len(set(instruments)):
            return False, "存在重复标的"
        
        # 检查操作合法性
        valid_actions = {"BUY", "SELL", "HOLD"}
        for s in self.signals:
            if s.action not in valid_actions:
                return False, f"非法操作: {s.action}"
            if s.target_weight < 0 or s.target_weight > 1:
                return False, f"权重越界: {s.target_weight}"
        
        return True, "OK"


# ============================================================================
#  信号导出器
# ============================================================================

class SignalExporter:
    """
    信号导出与推送引擎
    
    负责将模型预测转化为标准化信号批次，
    加密后推送至生产执行层。
    """
    
    def __init__(
        self,
        encryptor: Optional[Any] = None,
        signing_key: Optional[bytes] = None,
        target_url: str = "",
        timeout: int = 30,
    ):
        """
        Args:
            encryptor: AES256Encryptor 实例
            signing_key: HMAC 签名密钥
            target_url: 生产层 API 端点
            timeout: HTTP 超时
        """
        self.encryptor = encryptor
        self.signing_key = signing_key or hashlib.sha256(b"qlib-signal-signing").digest()
        self.target_url = target_url
        self.timeout = timeout
        self._logger = get_logger(__name__)
    
    def build_batch(
        self,
        predictions: Union[pd.DataFrame, pd.Series],
        weights: Optional[pd.Series] = None,
        model_name: str = "",
        model_version: str = "",
        approved_by: str = "",
        top_k: int = 30,
        min_score: Optional[float] = None,
    ) -> SignalBatch:
        """
        从预测得分构建信号批次
        
        Args:
            predictions: 预测得分 (行=日期, 列=标的 或 Series)
            weights: 组合权重 (可选)
            model_name: 模型名称
            model_version: 模型版本
            approved_by: 审批人
            top_k: 选取前 K 个标的
            min_score: 最低得分阈值
            
        Returns:
            SignalBatch
        """
        # 提取最新预测
        if isinstance(predictions, pd.DataFrame):
            latest = predictions.iloc[-1]
        else:
            latest = predictions
        
        # 排序选股
        sorted_scores = latest.sort_values(ascending=False)
        if min_score is not None:
            sorted_scores = sorted_scores[sorted_scores >= min_score]
        selected = sorted_scores.head(top_k)
        
        # 计算权重
        if weights is not None:
            if isinstance(weights, pd.DataFrame):
                w = weights.iloc[-1]
            else:
                w = weights
            target_weights = w.reindex(selected.index).fillna(0)
        else:
            # 等权
            target_weights = pd.Series(1.0 / len(selected), index=selected.index)
        
        # 归一化
        w_sum = target_weights.sum()
        if w_sum > 0:
            target_weights = target_weights / w_sum
        
        # 构建信号
        signals = []
        for inst in selected.index:
            signals.append(SignalEntry(
                instrument=str(inst),
                action="BUY",
                target_weight=float(target_weights.get(inst, 0)),
                score=float(selected[inst]),
            ))
        
        batch = SignalBatch(
            model_name=model_name,
            model_version=model_version,
            approved_by=approved_by,
            approved_at=datetime.now(timezone.utc).isoformat() if approved_by else "",
            signals=signals,
            metadata={
                "top_k": top_k,
                "min_score": min_score,
                "n_candidates": len(latest),
                "n_selected": len(selected),
            },
        )
        
        # 签名
        batch.signature = self._sign(batch)
        
        return batch
    
    def _sign(self, batch: SignalBatch) -> str:
        """HMAC 签名"""
        payload = json.dumps({
            "batch_id": batch.batch_id,
            "signals": [(s.instrument, s.action, s.target_weight) for s in batch.signals],
        }, sort_keys=True).encode("utf-8")
        
        return hmac.new(self.signing_key, payload, hashlib.sha256).hexdigest()
    
    def verify_signature(self, batch: SignalBatch) -> bool:
        """验证批次签名"""
        expected = self._sign(batch)
        return hmac.compare_digest(expected, batch.signature)
    
    def encrypt_batch(self, batch: SignalBatch) -> bytes:
        """加密信号批次"""
        payload = json.dumps(batch.to_payload(), ensure_ascii=False).encode("utf-8")
        
        if self.encryptor:
            return self.encryptor.encrypt(payload)
        
        self._logger.warning("无加密器，明文传输 (仅限开发环境)")
        return payload
    
    def decrypt_batch(self, ciphertext: bytes) -> SignalBatch:
        """解密信号批次"""
        if self.encryptor:
            plaintext = self.encryptor.decrypt(ciphertext)
        else:
            plaintext = ciphertext
        
        payload = json.loads(plaintext.decode("utf-8"))
        return SignalBatch.from_payload(payload)
    
    def push(
        self,
        batch: SignalBatch,
        target_url: Optional[str] = None,
        dry_run: bool = False,
    ) -> Dict[str, Any]:
        """
        推送信号批次至生产层
        
        Args:
            batch: 信号批次
            target_url: 目标 URL (覆盖默认)
            dry_run: 仅验证不实际推送
            
        Returns:
            {"success": bool, "message": str, "response": ...}
        """
        # 验证
        valid, msg = batch.validate()
        if not valid:
            return {"success": False, "message": f"批次验证失败: {msg}"}
        
        # 验证签名
        if not self.verify_signature(batch):
            return {"success": False, "message": "批次签名验证失败"}
        
        if dry_run:
            return {
                "success": True,
                "message": "DRY RUN - 未实际推送",
                "signals_count": len(batch.signals),
            }
        
        url = target_url or self.target_url
        if not url:
            return {"success": False, "message": "未配置目标 URL"}
        
        # 加密后推送
        try:
            import urllib.request
            
            ciphertext = self.encrypt_batch(batch)
            req = urllib.request.Request(
                url,
                data=ciphertext,
                headers={
                    "Content-Type": "application/octet-stream",
                    "X-Batch-ID": batch.batch_id,
                    "X-Signature": batch.signature,
                },
                method="POST",
            )
            
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                response_data = resp.read().decode("utf-8")
            
            self._logger.info(
                f"信号已推送 | batch={batch.batch_id} | "
                f"signals={len(batch.signals)} | "
                f"model={batch.model_name}"
            )
            
            return {
                "success": True,
                "message": "推送成功",
                "batch_id": batch.batch_id,
                "response": response_data,
            }
            
        except Exception as e:
            self._logger.error(f"信号推送失败: {e}")
            return {"success": False, "message": str(e)}
    
    def export_to_file(self, batch: SignalBatch, output_path: str, encrypt: bool = True) -> str:
        """导出信号批次到文件"""
        if encrypt:
            ciphertext = self.encrypt_batch(batch)
            mode = "wb"
            data = ciphertext
        else:
            data = json.dumps(batch.to_payload(), indent=2, ensure_ascii=False).encode("utf-8")
            mode = "w"
        
        with open(output_path, mode) as f:
            if isinstance(data, bytes):
                f.write(data) if mode == "wb" else f.write(data.decode("utf-8"))
            else:
                f.write(data)
        
        self._logger.info(f"信号已导出: {output_path}")
        return output_path


# ============================================================================
#  OMS 适配器 (订单管理系统)
# ============================================================================

@dataclass
class Order:
    """订单"""
    order_id: str
    instrument: str
    side: str           # BUY | SELL
    quantity: int
    price: Optional[float] = None
    order_type: str = "MKT"
    status: str = "PENDING"
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class OMSAdapter:
    """
    订单管理系统适配器基类
    
    为不同 OMS/EMS 系统提供统一接口。
    """
    
    def __init__(self, name: str = "generic_oms"):
        self.name = name
        self._orders: Dict[str, Order] = {}
        self._logger = get_logger(__name__)
    
    def submit_order(self, order: Order) -> bool:
        """提交订单"""
        self._orders[order.order_id] = order
        order.status = "SUBMITTED"
        self._logger.info(f"订单已提交: {order.order_id} | {order.side} {order.instrument} x{order.quantity}")
        return True
    
    def cancel_order(self, order_id: str) -> bool:
        """撤销订单"""
        if order_id in self._orders:
            self._orders[order_id].status = "CANCELLED"
            return True
        return False
    
    def get_order(self, order_id: str) -> Optional[Order]:
        return self._orders.get(order_id)
    
    def get_orders(self, status: Optional[str] = None) -> List[Order]:
        if status:
            return [o for o in self._orders.values() if o.status == status]
        return list(self._orders.values())
    
    def emergency_stop(self) -> int:
        """一键熔断：撤销所有未成交订单"""
        cancelled = 0
        for order in self._orders.values():
            if order.status in ("PENDING", "SUBMITTED"):
                order.status = "CANCELLED"
                cancelled += 1
        self._logger.warning(f"⚡ 一键熔断: 已撤销 {cancelled} 个订单")
        return cancelled
    
    def convert_signals(self, batch: SignalBatch, capital: float, prices: Dict[str, float]) -> List[Order]:
        """将信号批次转化为具体订单"""
        orders = []
        
        for i, signal in enumerate(batch.signals):
            price = prices.get(signal.instrument, 0)
            if price <= 0:
                continue
            
            alloc = capital * signal.target_weight
            quantity = int(alloc / price)
            
            if quantity <= 0:
                continue
            
            order = Order(
                order_id=f"{batch.batch_id}_{i:03d}",
                instrument=signal.instrument,
                side=signal.action,
                quantity=quantity,
                price=signal.limit_price or price,
                order_type=signal.order_type,
            )
            orders.append(order)
        
        return orders


# ============================================================================
#  生产环境安全网关
# ============================================================================

class ProductionGateway:
    """
    生产环境安全网关
    
    管理研究层与生产层的网络隔离与安全通信。
    """
    
    def __init__(
        self,
        encryptor: Optional[Any] = None,
        rbac: Optional[Any] = None,
        audit: Optional[Any] = None,
    ):
        self.encryptor = encryptor
        self.rbac = rbac
        self.audit = audit
        self._exporter: Optional[SignalExporter] = None
        self._logger = get_logger(__name__)
    
    def set_exporter(self, exporter: SignalExporter):
        self._exporter = exporter
    
    def approve_and_push(
        self,
        predictions: pd.DataFrame,
        approved_by: str,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        PM 审批并推送信号
        
        Args:
            predictions: 模型预测
            approved_by: PM 用户 ID
            **kwargs: 传递给 build_batch 的参数
            
        Returns:
            推送结果
        """
        # RBAC 检查
        if self.rbac and not self.rbac.can_push_signal(approved_by):
            msg = f"用户 {approved_by} 无权推送实盘信号"
            self._logger.error(msg)
            return {"success": False, "message": msg}
        
        if not self._exporter:
            return {"success": False, "message": "SignalExporter 未配置"}
        
        # 构建信号批次
        batch = self._exporter.build_batch(
            predictions=predictions,
            approved_by=approved_by,
            **kwargs,
        )
        
        # 审计日志
        if self.audit:
            self.audit.log(
                event_type="signal_push",
                user=approved_by,
                action="approve_and_push",
                resource="production_gateway",
                detail={
                    "batch_id": batch.batch_id,
                    "n_signals": len(batch.signals),
                    "model": batch.model_name,
                },
            )
        
        # 推送
        return self._exporter.push(batch)
    
    def emergency_shutdown(self, user_id: str, oms: Optional[OMSAdapter] = None) -> Dict[str, Any]:
        """
        一键熔断
        
        Args:
            user_id: 操作者
            oms: OMS 适配器实例
            
        Returns:
            熔断结果
        """
        # RBAC 检查
        if self.rbac and not self.rbac.can_emergency_stop(user_id):
            msg = f"用户 {user_id} 无权执行熔断"
            self._logger.error(msg)
            return {"success": False, "message": msg}
        
        # 审计日志
        if self.audit:
            self.audit.log(
                event_type="emergency_stop",
                user=user_id,
                action="circuit_breaker",
                resource="production_gateway",
                detail={"timestamp": datetime.now(timezone.utc).isoformat()},
            )
        
        result = {"success": True, "message": "熔断已执行"}
        
        if oms:
            cancelled = oms.emergency_stop()
            result["cancelled_orders"] = cancelled
        
        self._logger.critical(f"🔴 一键熔断已执行 by {user_id}")
        return result
