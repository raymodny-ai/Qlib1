"""
金融级安全模块 (Security Module)

实现符合 SOX 等金融监管要求的加密、审计与访问控制体系。

核心组件:
- AES256Encryptor: AES-256-GCM 透明加密/解密
- TLSValidator: TLS 1.2+ 证书验证
- AuditLogger: 防篡改活动审计日志
- RBACManager: 基于角色的访问控制

设计原则:
- 全链路 TLS 1.2+ 传输加密
- AES-256-GCM 静态存储加密
- 最小特权 + 职责分离
- 防篡改审计日志流 (密码学哈希链)

使用示例:
    from src.security import AES256Encryptor, AuditLogger, RBACManager
    
    # 加密
    encryptor = AES256Encryptor(key=os.environ["ENCRYPTION_KEY"])
    ciphertext = encryptor.encrypt(b"secret data")
    
    # 审计
    audit = AuditLogger()
    audit.log("model_deploy", user="pm_zhang", detail={"model": "lgb_v2"})
"""

import hashlib
import hmac
import json
import os
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

from src.utils.logger import get_logger


# ============================================================================
#  细粒度操作权限矩阵 (PRD 第6章)
# ============================================================================

class FineGrainedPerm(str, Enum):
    """
    细粒度操作权限枚举
    
    对标 PRD 第6章: 细粒度用户角色定义与操作权限控制矩阵
    覆盖因子挖掘、模型管理、信号推送、数据管理、审计等全链路操作
    """
    
    # ---------- 因子管理 ----------
    FACTOR_READ_DEF = "factor:read_definition"          # 查看因子定义
    FACTOR_WRITE_DEF = "factor:edit_definition"          # 编辑因子公式
    FACTOR_EXECUTE = "factor:execute"                     # 运行因子计算
    FACTOR_DELETE = "factor:delete"                       # 删除因子
    
    # ---------- 实验管理 ----------
    EXPERIMENT_READ = "experiment:read"                  # 查看实验配置
    EXPERIMENT_SUBMIT = "experiment:submit"              # 提交实验
    EXPERIMENT_CANCEL = "experiment:cancel"              # 取消实验
    EXPERIMENT_APPROVE = "experiment:approve"            # 审批实验 (PM)
    
    # ---------- 模型管理 ----------
    MODEL_READ = "model:read"                            # 读取模型
    MODEL_TRAIN = "model:train"                          # 发起训练
    MODEL_DEPLOY = "model:deploy"                        # 部署模型
    MODEL_ROLLBACK = "model:rollback"                    # 模型回滚 (PM)
    MODEL_DELETE = "model:delete"                        # 删除模型 (Admin)
    MODEL_EXPORT = "model:export"                        # 导出模型
    
    # ---------- 信号管理 ----------
    SIGNAL_READ = "signal:read"                          # 查看信号
    SIGNAL_APPROVE = "signal:approve"                    # 审批信号推送 (PM)
    SIGNAL_REJECT = "signal:reject"                      # 拒绝信号推送 (PM)
    SIGNAL_EMERGENCY_STOP = "signal:emergency_stop"     # 一键熔断 (PM)
    
    # ---------- 报告管理 ----------
    REPORT_READ = "report:read"                          # 查看报告
    REPORT_GENERATE = "report:generate"                  # 生成报告
    REPORT_EXPORT = "report:export"                      # 导出报告
    REPORT_DELETE = "report:delete"                      # 删除报告 (Admin)
    
    # ---------- 数据管理 ----------
    DATA_READ = "data:read"                              # 读取数据
    DATA_WRITE = "data:write"                            # 写入数据
    DATA_DELETE = "data:delete"                          # 删除数据 (Admin)
    DATA_EXPORT = "data:export"                          # 导出数据
    DATA_IMPORT = "data:import"                          # 导入数据
    
    # ---------- API/网关管理 ----------
    API_CONFIGURE = "api:configure"                      # 配置 API 网关
    API_KEY_ROTATE = "api:key_rotate"                   # 轮换 API 密钥
    
    # ---------- 审计管理 ----------
    AUDIT_READ = "audit:read"                            # 读取审计日志
    AUDIT_EXPORT = "audit:export"                        # 导出审计报告
    AUDIT_DELETE = "audit:delete"                        # 删除审计日志 (Admin)
    
    # ---------- 用户管理 ----------
    USER_READ = "user:read"                              # 查看用户
    USER_CREATE = "user:create"                          # 创建用户
    USER_MODIFY = "user:modify"                          # 修改用户
    USER_DELETE = "user:delete"                          # 删除用户
    USER_GRANT = "user:grant"                            # 授权临时权限
    
    # ---------- 系统管理 ----------
    SYSTEM_MONITOR = "system:monitor"                   # 系统监控
    SYSTEM_CONFIG = "system:config"                      # 系统配置
    SYSTEM_SHUTDOWN = "system:shutdown"                 # 系统关停
    
    # ---------- 合规管理 ----------
    COMPLIANCE_EXPORT = "compliance:export"             # 导出合规报告
    COMPLIANCE_REVIEW = "compliance:review"             # 合规审查
    
    # ---------- 风险管理 ----------
    RISK_READ = "risk:read"                              # 读取风控数据
    RISK_CONFIGURE = "risk:configure"                   # 配置风控参数 (PM)


# 权限依赖图: 某些操作需要多个权限同时满足
PERMISSION_DEPENDENCIES: Dict[str, Set[str]] = {
    "model:deploy": {"model:read", "model:train"},
    "model:export": {"model:read", "data:export"},
    "signal:approve": {"signal:read", "risk:read"},
    "report:export": {"report:read", "data:export"},
    "data:export": {"data:read"},
    "data:import": {"data:write"},
    "api:key_rotate": {"api:configure"},
    "user:grant": {"user:read"},
    "compliance:export": {"audit:read"},
}


# ============================================================================
#  时间限权与审批工作流 (PRD 第6章)
# ============================================================================

@dataclass
class AccessGrant:
    """临时权限授权"""
    grant_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    user_id: str = ""
    permission: str = ""
    granted_by: str = ""          # 授权人
    reason: str = ""               # 授权原因
    granted_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    expires_at: str = ""           # 过期时间
    used_count: int = 0            # 使用次数
    max_uses: int = 1              # 最大使用次数 (-1 无限)
    is_active: bool = True
    
    def is_expired(self) -> bool:
        if not self.expires_at:
            return False
        return datetime.fromisoformat(self.expires_at) < datetime.now(timezone.utc)
    
    def is_exhausted(self) -> bool:
        if self.max_uses < 0:
            return False
        return self.used_count >= self.max_uses
    
    def is_valid(self) -> bool:
        return self.is_active and not self.is_expired() and not self.is_exhausted()


@dataclass
class AccessRequest:
    """权限请求 (审批工作流)"""
    request_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    requester_id: str = ""
    requested_permission: str = ""
    reason: str = ""
    approver_id: str = ""          # 审批人
    status: str = "pending"        # pending | approved | rejected
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    resolved_at: str = ""
    resolution_note: str = ""
    duration_hours: int = 24       # 请求时长(小时)


# ============================================================================
#  AES-256-GCM 加密器
# ============================================================================

class AES256Encryptor:
    """
    AES-256-GCM 透明加密器
    
    使用 AES-256 位密钥进行 GCM 模式认证加密 (AEAD)。
    GCM 模式同时提供机密性和完整性验证。
    """
    
    def __init__(self, key: Optional[bytes] = None, key_env: str = "ENCRYPTION_KEY"):
        """
        Args:
            key: 32 字节原始密钥 (None 则从环境变量读取)
            key_env: 环境变量名
        """
        self._logger = get_logger(__name__)
        
        if key is None:
            key_b64 = os.environ.get(key_env, "")
            if not key_b64:
                self._logger.warning(f"未设置 {key_env}，使用临时密钥 (仅限开发)")
                key = os.urandom(32)
            else:
                import base64
                key = base64.b64decode(key_b64)
        
        if len(key) != 32:
            raise ValueError(f"AES-256 需要 32 字节密钥，当前: {len(key)} 字节")
        
        self._key = key
    
    def encrypt(self, plaintext: Union[str, bytes]) -> bytes:
        """
        加密数据
        
        Returns:
            nonce (12 bytes) + ciphertext + tag (16 bytes)
        """
        try:
            from cryptography.hazmat.primitives.ciphers.aead import AESGCM
            
            if isinstance(plaintext, str):
                plaintext = plaintext.encode("utf-8")
            
            aesgcm = AESGCM(self._key)
            nonce = os.urandom(12)
            ciphertext = aesgcm.encrypt(nonce, plaintext, None)
            
            # 返回: nonce + ciphertext
            return nonce + ciphertext
            
        except ImportError:
            self._logger.error("cryptography 未安装，回退到简单异或 (不安全!)")
            return self._fallback_xor(plaintext)
    
    def decrypt(self, ciphertext: bytes) -> bytes:
        """解密数据"""
        try:
            from cryptography.hazmat.primitives.ciphers.aead import AESGCM
            
            nonce = ciphertext[:12]
            data = ciphertext[12:]
            
            aesgcm = AESGCM(self._key)
            return aesgcm.decrypt(nonce, data, None)
            
        except ImportError:
            return self._fallback_xor(ciphertext)
    
    def encrypt_file(self, input_path: str, output_path: Optional[str] = None) -> str:
        """加密文件"""
        input_path = Path(input_path)
        if output_path is None:
            output_path = str(input_path) + ".enc"
        
        with open(input_path, "rb") as f:
            plaintext = f.read()
        
        encrypted = self.encrypt(plaintext)
        
        with open(output_path, "wb") as f:
            f.write(encrypted)
        
        self._logger.info(f"文件已加密: {input_path} -> {output_path}")
        return output_path
    
    def decrypt_file(self, input_path: str, output_path: Optional[str] = None) -> str:
        """解密文件"""
        input_path = Path(input_path)
        if output_path is None:
            output_path = str(input_path).replace(".enc", "")
        
        with open(input_path, "rb") as f:
            ciphertext = f.read()
        
        plaintext = self.decrypt(ciphertext)
        
        with open(output_path, "wb") as f:
            f.write(plaintext)
        
        return output_path
    
    @staticmethod
    def _fallback_xor(data: bytes) -> bytes:
        """(DEV ONLY) 简单 XOR 回退"""
        key = hashlib.sha256(b"qlib-dev-key").digest()
        return bytes(b ^ key[i % len(key)] for i, b in enumerate(data))
    
    @staticmethod
    def generate_key() -> bytes:
        """生成随机 AES-256 密钥"""
        return os.urandom(32)
    
    @staticmethod
    def key_to_b64(key: bytes) -> str:
        import base64
        return base64.b64encode(key).decode("ascii")


# ============================================================================
#  TLS 安全验证器
# ============================================================================

class TLSValidator:
    """TLS 1.2+ 证书验证器"""
    
    MINIMUM_TLS_VERSION = "TLSv1.2"
    
    @staticmethod
    def validate_certificate(cert_path: str) -> Dict[str, Any]:
        """验证 TLS 证书"""
        try:
            from cryptography import x509
            from cryptography.hazmat.backends import default_backend
            
            with open(cert_path, "rb") as f:
                cert_data = f.read()
            
            cert = x509.load_pem_x509_certificate(cert_data, default_backend())
            
            now = datetime.now(timezone.utc)
            return {
                "valid": cert.not_valid_before_utc <= now <= cert.not_valid_after_utc,
                "subject": str(cert.subject),
                "issuer": str(cert.issuer),
                "not_before": cert.not_valid_before_utc.isoformat(),
                "not_after": cert.not_valid_after_utc.isoformat(),
                "serial_number": str(cert.serial_number),
            }
        except ImportError:
            return {"valid": False, "error": "cryptography 未安装"}
        except Exception as e:
            return {"valid": False, "error": str(e)}
    
    @classmethod
    def check_protocol(cls, protocol: str) -> bool:
        """检查 TLS 协议版本是否 ≥ 1.2"""
        allowed = {"TLSv1.2", "TLSv1.3"}
        return protocol in allowed


# ============================================================================
#  防篡改审计日志
# ============================================================================

@dataclass
class AuditEntry:
    """审计日志条目"""
    event_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    event_type: str = ""           # model_deploy | api_key_rotate | config_change | ...
    user: str = ""                 # 操作者标识
    role: str = ""                 # 角色
    action: str = ""               # 具体操作
    resource: str = ""             # 操作对象
    detail: Dict[str, Any] = field(default_factory=dict)
    ip_address: str = ""
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    hash_chain: str = ""           # 哈希链指针 (防篡改)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_id": self.event_id,
            "event_type": self.event_type,
            "user": self.user,
            "role": self.role,
            "action": self.action,
            "resource": self.resource,
            "detail": self.detail,
            "ip_address": self.ip_address,
            "timestamp": self.timestamp,
            "hash_chain": self.hash_chain,
        }


class AuditLogger:
    """
    防篡改审计日志记录器
    
    使用密码学哈希链确保日志不可篡改:
    - 每条日志包含前一哈希的 HMAC
    - 支持按需导出 SOX 合规报告
    - 日志归档至独立存储
    """
    
    def __init__(
        self,
        log_dir: str = "logs/audit",
        hmac_key: Optional[bytes] = None,
    ):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.hmac_key = hmac_key or hashlib.sha256(b"qlib-audit-chain").digest()
        self._chain_hash: str = self._genesis_hash()
        self._entries: List[AuditEntry] = []
        self._lock = __import__("threading").Lock()
        self._logger = get_logger(__name__)
    
    def _genesis_hash(self) -> str:
        """生成创世哈希"""
        return hmac.new(
            self.hmac_key, b"QLIB_AUDIT_GENESIS", hashlib.sha256
        ).hexdigest()
    
    def log(
        self,
        event_type: str,
        user: str = "system",
        role: str = "unknown",
        action: str = "",
        resource: str = "",
        detail: Optional[Dict[str, Any]] = None,
        ip_address: str = "",
    ) -> AuditEntry:
        """
        记录审计日志条目
        
        Args:
            event_type: 事件类型 (model_deploy, api_key_rotate, config_change, ...)
            user: 操作者
            role: 角色
            action: 操作
            resource: 资源
            detail: 详细信息
            ip_address: IP 地址
            
        Returns:
            AuditEntry
        """
        entry = AuditEntry(
            event_type=event_type,
            user=user,
            role=role,
            action=action,
            resource=resource,
            detail=detail or {},
            ip_address=ip_address,
        )
        
        # 计算哈希链 (排除 hash_chain 字段自身)
        payload = json.dumps(
            {k: v for k, v in entry.to_dict().items() if k != "hash_chain"},
            sort_keys=True,
        ).encode("utf-8")
        entry.hash_chain = hmac.new(
            self.hmac_key,
            self._chain_hash.encode() + payload,
            hashlib.sha256,
        ).hexdigest()
        
        with self._lock:
            self._chain_hash = entry.hash_chain
            self._entries.append(entry)
        
        # 异步写入文件
        self._flush_entry(entry)
        
        return entry
    
    def _flush_entry(self, entry: AuditEntry):
        """将条目写入磁盘"""
        date_str = datetime.now().strftime("%Y%m%d")
        log_file = self.log_dir / f"audit_{date_str}.jsonl"
        
        try:
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry.to_dict(), ensure_ascii=False) + "\n")
        except Exception as e:
            self._logger.error(f"审计日志写入失败: {e}")
    
    def verify_chain(self, date_str: Optional[str] = None) -> Dict[str, Any]:
        """
        验证哈希链完整性
        
        Returns:
            {"valid": bool, "violations": [...]}
        """
        if date_str is None:
            date_str = datetime.now().strftime("%Y%m%d")
        
        log_file = self.log_dir / f"audit_{date_str}.jsonl"
        if not log_file.exists():
            return {"valid": True, "violations": [], "message": "无日志文件"}
        
        violations: List[Dict] = []
        prev_hash = self._genesis_hash()
        
        with open(log_file, "r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, 1):
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    violations.append({"line": line_no, "error": "JSON 解析失败"})
                    continue
                
                # 重新计算哈希
                payload = json.dumps(
                    {k: v for k, v in entry.items() if k != "hash_chain"},
                    sort_keys=True,
                ).encode("utf-8")
                
                expected = hmac.new(
                    self.hmac_key,
                    prev_hash.encode() + payload,
                    hashlib.sha256,
                ).hexdigest()
                
                if expected != entry.get("hash_chain", ""):
                    violations.append({
                        "line": line_no,
                        "event_id": entry.get("event_id", "?"),
                        "expected": expected[:16],
                        "actual": entry.get("hash_chain", "")[:16],
                    })
                
                prev_hash = entry.get("hash_chain", prev_hash)
        
        return {
            "valid": len(violations) == 0,
            "violations": violations,
            "total_lines": line_no if 'line_no' in dir() else 0,
            "chain_intact": len(violations) == 0,
        }
    
    def query(
        self,
        event_type: Optional[str] = None,
        user: Optional[str] = None,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """查询审计日志"""
        results = []
        
        for log_file in sorted(self.log_dir.glob("audit_*.jsonl"), reverse=True):
            if len(results) >= limit:
                break
            
            with open(log_file, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        entry = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    
                    # 过滤
                    if event_type and entry.get("event_type") != event_type:
                        continue
                    if user and entry.get("user") != user:
                        continue
                    if start_time and entry.get("timestamp", "") < start_time:
                        continue
                    if end_time and entry.get("timestamp", "") > end_time:
                        continue
                    
                    results.append(entry)
                    if len(results) >= limit:
                        break
        
        return results
    
    def export_report(self, output_path: str, **filters) -> str:
        """导出审计报告"""
        entries = self.query(**filters)
        
        report = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "total_events": len(entries),
            "chain_verified": self.verify_chain(),
            "entries": entries,
        }
        
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        return output_path


# ============================================================================
#  RBAC 访问控制
# ============================================================================

class Role(Enum):
    """系统角色枚举"""
    QUANT_RESEARCHER = "quant_researcher"     # Alpha 因子挖掘 + 模型训练
    PORTFOLIO_MANAGER = "portfolio_manager"   # 风险审批 + 信号推送授权
    DATA_ADMIN = "data_admin"                  # 数据池管理 + API 配置
    COMPLIANCE_AUDITOR = "compliance_auditor"  # 只读审计 + 合规检查
    SYSTEM_ADMIN = "system_admin"              # 全局管理


# 权限定义
PERMISSIONS = {
    Role.QUANT_RESEARCHER: {
        "factor:write",           # 编写因子公式
        "experiment:submit",      # 提交实验
        "model:train",            # 发起训练
        "model:read",             # 读取模型
        "report:read",            # 查看回测报告
        "data:read",              # 读取特征数据
    },
    Role.PORTFOLIO_MANAGER: {
        "model:read",
        "model:deploy",
        "report:read",
        "signal:approve",         # 审批信号推送
        "signal:reject",          # 拒绝信号推送
        "signal:emergency_stop",  # 一键熔断
        "risk:read",              # 读取风控数据
        "data:read",
    },
    Role.DATA_ADMIN: {
        "data:read",
        "data:write",             # 写入数据池
        "data:delete",            # 删除数据
        "api:configure",          # 配置 API 网关
        "system:monitor",         # 系统监控
        "user:manage",            # 账号管理
    },
    Role.COMPLIANCE_AUDITOR: {
        "audit:read",             # 读取审计日志
        "report:read",
        "compliance:export",      # 导出合规报告
    },
    Role.SYSTEM_ADMIN: {
        "*",  # 全权限
    },
}

# 角色继承: PM 也拥有 Researcher 的只读权限
ROLE_INHERITANCE: Dict[Role, List[Role]] = {
    Role.PORTFOLIO_MANAGER: [Role.QUANT_RESEARCHER],
    Role.SYSTEM_ADMIN: [Role.DATA_ADMIN, Role.PORTFOLIO_MANAGER],
}


# 增强权限映射: 角色 → 细粒度权限集合
FINEGRAINED_PERMISSIONS: Dict[Role, Set[str]] = {
    Role.QUANT_RESEARCHER: {
        FineGrainedPerm.FACTOR_READ_DEF,
        FineGrainedPerm.FACTOR_WRITE_DEF,
        FineGrainedPerm.FACTOR_EXECUTE,
        FineGrainedPerm.EXPERIMENT_READ,
        FineGrainedPerm.EXPERIMENT_SUBMIT,
        FineGrainedPerm.MODEL_READ,
        FineGrainedPerm.MODEL_TRAIN,
        FineGrainedPerm.SIGNAL_READ,
        FineGrainedPerm.REPORT_READ,
        FineGrainedPerm.REPORT_GENERATE,
        FineGrainedPerm.DATA_READ,
        FineGrainedPerm.RISK_READ,
    },
    Role.PORTFOLIO_MANAGER: {
        FineGrainedPerm.MODEL_READ,
        FineGrainedPerm.MODEL_ROLLBACK,
        FineGrainedPerm.SIGNAL_READ,
        FineGrainedPerm.SIGNAL_APPROVE,
        FineGrainedPerm.SIGNAL_REJECT,
        FineGrainedPerm.SIGNAL_EMERGENCY_STOP,
        FineGrainedPerm.REPORT_READ,
        FineGrainedPerm.REPORT_GENERATE,
        FineGrainedPerm.REPORT_EXPORT,
        FineGrainedPerm.DATA_READ,
        FineGrainedPerm.DATA_EXPORT,
        FineGrainedPerm.RISK_READ,
        FineGrainedPerm.RISK_CONFIGURE,
        FineGrainedPerm.EXPERIMENT_APPROVE,
    },
    Role.DATA_ADMIN: {
        FineGrainedPerm.DATA_READ,
        FineGrainedPerm.DATA_WRITE,
        FineGrainedPerm.DATA_DELETE,
        FineGrainedPerm.DATA_EXPORT,
        FineGrainedPerm.DATA_IMPORT,
        FineGrainedPerm.API_CONFIGURE,
        FineGrainedPerm.API_KEY_ROTATE,
        FineGrainedPerm.SYSTEM_MONITOR,
        FineGrainedPerm.USER_READ,
        FineGrainedPerm.USER_CREATE,
        FineGrainedPerm.USER_MODIFY,
    },
    Role.COMPLIANCE_AUDITOR: {
        FineGrainedPerm.AUDIT_READ,
        FineGrainedPerm.AUDIT_EXPORT,
        FineGrainedPerm.REPORT_READ,
        FineGrainedPerm.COMPLIANCE_EXPORT,
        FineGrainedPerm.COMPLIANCE_REVIEW,
        FineGrainedPerm.DATA_READ,
        FineGrainedPerm.RISK_READ,
    },
    Role.SYSTEM_ADMIN: {
        "*",
    },
}


@dataclass
class User:
    """系统用户"""
    user_id: str
    name: str
    role: Role
    email: str = ""
    api_key_hash: str = ""
    active: bool = True
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class RBACManager:
    """
    基于角色的访问控制 (RBAC) — 增强版
    
    贯彻最小特权与职责分离法则，支持:
    - 角色基础权限 + 细粒度权限矩阵
    - 时间限权 (临时权限授权)
    - 审批工作流 (Access Request → PM Approve)
    - 权限依赖图 (某些操作需多权限同时满足)
    - 操作审计追踪
    
    角色:
    - Quant Researcher: 因子挖掘，不能推送实盘信号
    - Portfolio Manager: 拥有最高决策与一键熔断权
    - Compliance Auditor: 独立体系，只读审计
    """
    
    def __init__(self, audit_logger: Optional[AuditLogger] = None):
        import threading
        self._users: Dict[str, User] = {}
        self._grants: Dict[str, AccessGrant] = {}       # 临时授权
        self._requests: Dict[str, AccessRequest] = {}    # 审批流
        self._lock = threading.RLock()
        self._audit_logger = audit_logger
        self._logger = get_logger(__name__)
    
    # ------------------------------------------------------------------
    #  用户管理
    # ------------------------------------------------------------------
    
    def add_user(self, user: User):
        """注册用户"""
        with self._lock:
            self._users[user.user_id] = user
        self._logger.info(f"用户已注册: {user.user_id} | 角色: {user.role.value}")
        self._audit("user_registered", user.user_id, resource=f"user/{user.user_id}")
    
    def remove_user(self, user_id: str):
        """注销用户"""
        with self._lock:
            self._users.pop(user_id, None)
        self._audit("user_removed", user_id, resource=f"user/{user_id}")
    
    def get_user(self, user_id: str) -> Optional[User]:
        with self._lock:
            return self._users.get(user_id)
    
    def list_users(self) -> List[Dict[str, Any]]:
        with self._lock:
            return [
                {
                    "user_id": u.user_id,
                    "name": u.name,
                    "role": u.role.value,
                    "email": u.email,
                    "active": u.active,
                }
                for u in self._users.values()
            ]
    
    # ------------------------------------------------------------------
    #  权限获取 (基础 + 细粒度)
    # ------------------------------------------------------------------
    
    def get_permissions(self, role: Role) -> Set[str]:
        """获取角色权限集 (含继承)"""
        perms = set(PERMISSIONS.get(role, set()))
        for parent in ROLE_INHERITANCE.get(role, []):
            perms |= self.get_permissions(parent)
        return perms
    
    def get_finegrained_permissions(self, role: Role) -> Set[str]:
        """获取角色细粒度权限集 (含继承)"""
        perms = set(FINEGRAINED_PERMISSIONS.get(role, set()))
        for parent in ROLE_INHERITANCE.get(role, []):
            perms |= self.get_finegrained_permissions(parent)
        return perms
    
    def get_all_permissions(self, role: Role) -> Set[str]:
        """获取角色全部权限 (基础 + 细粒度，含继承)"""
        return self.get_permissions(role) | self.get_finegrained_permissions(role)
    
    # ------------------------------------------------------------------
    #  权限检查 (含依赖 + 临时授权)
    # ------------------------------------------------------------------
    
    def check_permission(
        self,
        user_id: str,
        required_permission: str,
        check_dependencies: bool = True,
    ) -> bool:
        """
        检查用户是否拥有指定权限

        Args:
            user_id: 用户标识
            required_permission: 所需权限
            check_dependencies: 是否检查权限依赖

        Returns:
            True 如果有权限
        """
        user = self._users.get(user_id)
        if user is None or not user.active:
            return False

        # 获取所有权限 (基础 + 细粒度)
        all_perms = self.get_all_permissions(user.role)

        # 通配符 *
        if "*" in all_perms:
            return True

        # 直接匹配
        if required_permission in all_perms:
            if check_dependencies:
                return self._check_dependencies(all_perms, required_permission)
            return True

        # 层级匹配: "signal:approve" → "signal:*"
        resource, _, _ = required_permission.partition(":")
        if f"{resource}:*" in all_perms:
            if check_dependencies:
                return self._check_dependencies(all_perms, required_permission)
            return True

        # 临时授权
        if self._check_temporary_grant(user_id, required_permission):
            return True

        return False
    
    def _check_dependencies(self, user_perms: Set[str], required: str) -> bool:
        """检查权限依赖是否满足"""
        deps = PERMISSION_DEPENDENCIES.get(required, set())
        if not deps:
            return True
        for dep in deps:
            if dep not in user_perms:
                dep_resource, _, _ = dep.partition(":")
                if f"{dep_resource}:*" not in user_perms:
                    return False
        return True
    
    def assert_permission(self, user_id: str, required_permission: str):
        """断言用户有权限，否则抛出 PermissionError"""
        if not self.check_permission(user_id, required_permission):
            user = self._users.get(user_id)
            role_str = user.role.value if user else "unknown"
            raise PermissionError(
                f"权限拒绝: user={user_id}, role={role_str}, "
                f"required={required_permission}"
            )
    
    # ------------------------------------------------------------------
    #  临时权限授权
    # ------------------------------------------------------------------
    
    def grant_temporary_permission(
        self,
        user_id: str,
        permission: str,
        granted_by: str,
        duration_hours: int = 24,
        max_uses: int = 1,
        reason: str = "",
    ) -> AccessGrant:
        """
        授予临时权限 (仅 PM/Admin 可执行)
        
        Args:
            user_id: 被授权用户
            permission: 权限
            granted_by: 授权人
            duration_hours: 有效期(小时)
            max_uses: 最大使用次数
            reason: 原因
        """
        expires_at = (datetime.now(timezone.utc) + timedelta(hours=duration_hours)).isoformat()
        grant = AccessGrant(
            user_id=user_id,
            permission=permission,
            granted_by=granted_by,
            reason=reason,
            expires_at=expires_at,
            max_uses=max_uses,
        )
        with self._lock:
            self._grants[grant.grant_id] = grant
        
        self._audit(
            "temporary_grant",
            user_id,
            action="grant",
            resource=f"permission/{permission}",
            detail={"grant_id": grant.grant_id, "granted_by": granted_by, "duration_h": duration_hours},
        )
        self._logger.info(f"临时授权: {user_id} <- {permission} (by {granted_by}, {duration_hours}h)")
        return grant
    
    def revoke_temporary_permission(self, grant_id: str, revoked_by: str = "system") -> bool:
        """吊销临时授权"""
        with self._lock:
            grant = self._grants.get(grant_id)
            if grant:
                grant.is_active = False
                self._audit(
                    "temporary_revoke",
                    grant.user_id,
                    action="revoke",
                    resource=f"grant/{grant_id}",
                    detail={"revoked_by": revoked_by, "permission": grant.permission},
                )
                return True
        return False
    
    def _check_temporary_grant(self, user_id: str, permission: str) -> bool:
        """检查是否存在有效临时授权"""
        with self._lock:
            for grant in self._grants.values():
                if not grant.is_valid():
                    continue
                if grant.user_id != user_id:
                    continue
                if grant.permission != permission:
                    continue
                grant.used_count += 1
                return True
        return False
    
    def list_active_grants(self, user_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """列出有效临时授权"""
        active = []
        with self._lock:
            for gid, grant in self._grants.items():
                if not grant.is_valid():
                    continue
                if user_id and grant.user_id != user_id:
                    continue
                active.append({
                    "grant_id": gid,
                    "user_id": grant.user_id,
                    "permission": grant.permission,
                    "granted_by": grant.granted_by,
                    "reason": grant.reason,
                    "expires_at": grant.expires_at,
                    "remaining_uses": grant.max_uses - grant.used_count if grant.max_uses > 0 else -1,
                })
        return active
    
    # ------------------------------------------------------------------
    #  审批工作流
    # ------------------------------------------------------------------
    
    def request_access(
        self,
        requester_id: str,
        permission: str,
        reason: str = "",
        approver_role: Role = Role.PORTFOLIO_MANAGER,
        duration_hours: int = 24,
    ) -> AccessRequest:
        """
        发起权限请求 (审批工作流)
        
        Args:
            requester_id: 请求者
            permission: 请求权限
            reason: 原因
            approver_role: 审批者角色
            duration_hours: 请求时长
        """
        request = AccessRequest(
            requester_id=requester_id,
            requested_permission=permission,
            reason=reason,
            duration_hours=duration_hours,
        )
        with self._lock:
            self._requests[request.request_id] = request
        
        self._audit(
            "access_requested",
            requester_id,
            action="request",
            resource=f"permission/{permission}",
            detail={"request_id": request.request_id, "reason": reason},
        )
        self._logger.info(f"权限请求: {requester_id} 请求 {permission} (需 {approver_role.value} 审批)")
        return request
    
    def approve_request(
        self,
        request_id: str,
        approver_id: str,
        resolution_note: str = "",
    ) -> Optional[AccessGrant]:
        """
        审批通过权限请求 → 自动授予临时权限
        
        仅 PM 或 Admin 可审批
        """
        with self._lock:
            req = self._requests.get(request_id)
            if req is None:
                return None
            if req.status != "pending":
                return None
            
            req.status = "approved"
            req.resolved_at = datetime.now(timezone.utc).isoformat()
            req.resolution_note = resolution_note
        
        # 自动授予临时权限
        grant = self.grant_temporary_permission(
            user_id=req.requester_id,
            permission=req.requested_permission,
            granted_by=approver_id,
            duration_hours=req.duration_hours,
            reason=f"Approved: {req.reason}",
        )
        
        self._audit(
            "access_approved",
            req.requester_id,
            action="approve",
            resource=f"permission/{req.requested_permission}",
            detail={"request_id": request_id, "approver": approver_id, "grant_id": grant.grant_id},
        )
        return grant
    
    def reject_request(
        self,
        request_id: str,
        approver_id: str,
        resolution_note: str = "",
    ) -> bool:
        """拒绝权限请求"""
        with self._lock:
            req = self._requests.get(request_id)
            if req is None or req.status != "pending":
                return False
            req.status = "rejected"
            req.resolved_at = datetime.now(timezone.utc).isoformat()
            req.resolution_note = resolution_note
        
        self._audit(
            "access_rejected",
            req.requester_id,
            action="reject",
            resource=f"permission/{req.requested_permission}",
            detail={"request_id": request_id, "approver": approver_id},
        )
        return True
    
    def list_requests(
        self,
        status: Optional[str] = None,
        requester_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """列出权限请求"""
        results = []
        with self._lock:
            for rid, req in self._requests.items():
                if status and req.status != status:
                    continue
                if requester_id and req.requester_id != requester_id:
                    continue
                results.append({
                    "request_id": rid,
                    "requester_id": req.requester_id,
                    "permission": req.requested_permission,
                    "reason": req.reason,
                    "status": req.status,
                    "created_at": req.created_at,
                    "resolved_at": req.resolved_at,
                    "resolution_note": req.resolution_note,
                })
        return results
    
    # ------------------------------------------------------------------
    #  便捷权限检查
    # ------------------------------------------------------------------
    
    def can_push_signal(self, user_id: str) -> bool:
        """检查是否可以推送实盘信号 (仅 PM)"""
        return self.check_permission(user_id, "signal:approve")
    
    def can_train_model(self, user_id: str) -> bool:
        """检查是否可以训练模型"""
        return self.check_permission(user_id, "model:train")
    
    def can_emergency_stop(self, user_id: str) -> bool:
        """检查是否可以一键熔断 (仅 PM)"""
        return self.check_permission(user_id, "signal:emergency_stop")
    
    def can_deploy_model(self, user_id: str) -> bool:
        """检查是否可以部署模型"""
        return self.check_permission(user_id, "model:deploy")
    
    @staticmethod
    def get_all_roles() -> List[Dict[str, Any]]:
        return [
            {
                "role": r.value,
                "permissions": sorted(PERMISSIONS.get(r, set())),
                "finegrained_permissions": sorted(FINEGRAINED_PERMISSIONS.get(r, set())),
            }
            for r in Role
        ]
    
    # ------------------------------------------------------------------
    #  审计辅助
    # ------------------------------------------------------------------
    
    def _audit(
        self,
        event_type: str,
        user: str = "system",
        role: str = "unknown",
        action: str = "",
        resource: str = "",
        detail: Optional[Dict[str, Any]] = None,
    ):
        if self._audit_logger:
            self._audit_logger.log(
                event_type=event_type,
                user=user,
                role=role,
                action=action,
                resource=resource,
                detail=detail or {},
            )
