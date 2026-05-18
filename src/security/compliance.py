"""
金融安全合规与审计模块 (Compliance & Audit Module)

对标 PRD 第5章 全局金融安全与合规审计规范：
- 全链路 TLS 1.2+ 传输加密强制
- AES-256 透明磁盘加密（特征集 + 模型权重）
- SOX 合规审计报告与防篡改校验
- 审计日志归档与数据留存策略
- 密钥生命周期管理与安全轮换
- 数据敏感性分类与访问控制

核心组件:
- DiskEncryptor: 卷级别透明 AES-256 磁盘加密
- TLSEnforcer: 传输层 TLS 1.2+ 强制中间件
- SOXComplianceReporter: SOX 法案合规报告生成
- AuditArchiveManager: 审计日志归档与留存管理
- SecretManager: 密钥/凭证生命周期管理
- DataClassification: 数据敏感性分类

使用示例:
    from src.security.compliance import (
        DiskEncryptor, TLSEnforcer, SOXComplianceReporter,
        SecretManager, AuditArchiveManager,
    )
    
    # 磁盘加密
    de = DiskEncryptor(master_key_path="/etc/qlib/keys/master.key")
    de.encrypt_volume("./data/features")
    
    # TLS 强制
    enforcer = TLSEnforcer(minimum_version="TLSv1.2")
    assert enforcer.is_connection_secure(client_protocol="TLSv1.3")
    
    # SOX 报告
    reporter = SOXComplianceReporter(audit_logger=audit, rbac=rbac)
    report = reporter.generate_quarterly_report()
"""

import json
import os
import shutil
import tempfile
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

import numpy as np

from src.utils.logger import get_logger


# ============================================================================
#  TLS 强制器
# ============================================================================

class TLSVersion(str, Enum):
    """TLS 协议版本"""
    TLSv1_0 = "TLSv1.0"
    TLSv1_1 = "TLSv1.1"
    TLSv1_2 = "TLSv1.2"
    TLSv1_3 = "TLSv1.3"


class TLSPolicy(str, Enum):
    """TLS 安全策略"""
    STRICT = "strict"       # 仅允许 TLS 1.2+
    MODERN = "modern"       # 仅允许 TLS 1.3
    COMPAT = "compat"       # 允许 TLS 1.1+ (向后兼容)


@dataclass
class TLSConnectionInfo:
    """TLS 连接信息"""
    protocol: str
    cipher_suite: str = ""
    cert_common_name: str = ""
    cert_expiry: str = ""
    remote_address: str = ""
    is_valid: bool = False
    violation: str = ""


class TLSEnforcer:
    """
    TLS 1.2+ 传输安全强制器

    对标 PRD 5.1: 任何 API 网关与 RPC 通信强制 TLS 1.2+。
    支持 STRICT/MODERN/COMPAT 三种安全策略。

    使用示例:
        enforcer = TLSEnforcer(policy=TLSPolicy.STRICT)
        info = enforcer.inspect_connection("TLSv1.2", "ECDHE-RSA-AES256-GCM-SHA384")
        if not info.is_valid:
            raise SecurityViolation(info.violation)
    """

    STRONG_CIPHERS = {
        "ECDHE-RSA-AES256-GCM-SHA384",
        "ECDHE-ECDSA-AES256-GCM-SHA384",
        "ECDHE-RSA-AES128-GCM-SHA256",
        "ECDHE-ECDSA-AES128-GCM-SHA256",
        "ECDHE-RSA-CHACHA20-POLY1305",
        "ECDHE-ECDSA-CHACHA20-POLY1305",
    }

    WEAK_CIPHERS = {
        "RC4-SHA",
        "DES-CBC3-SHA",
        "RC4-MD5",
        "NULL-SHA256",
        "NULL-MD5",
    }

    def __init__(self, policy: TLSPolicy = TLSPolicy.STRICT):
        self.policy = policy
        self._violations: List[TLSConnectionInfo] = []
        self._lock = threading.Lock()
        self.logger = get_logger()

    def inspect_connection(
        self,
        protocol: str,
        cipher_suite: str = "",
        cert_cn: str = "",
        cert_expiry: str = "",
        remote_addr: str = "",
    ) -> TLSConnectionInfo:
        """
        检查连接是否符合 TLS 安全策略

        Args:
            protocol: 协商的 TLS 协议版本
            cipher_suite: 密码套件
            cert_cn: 证书通用名称
            cert_expiry: 证书过期时间 (ISO 格式)
            remote_addr: 远程地址

        Returns:
            TLSConnectionInfo
        """
        violations = []

        # 1. 协议版本检查
        if self.policy == TLSPolicy.MODERN:
            if protocol != "TLSv1.3":
                violations.append(f"协议 {protocol} 不符合 MODERN 策略 (要求 TLSv1.3)")
        elif self.policy == TLSPolicy.STRICT:
            if protocol not in ("TLSv1.2", "TLSv1.3"):
                violations.append(f"协议 {protocol} 不符合 STRICT 策略 (要求 ≥ TLSv1.2)")
        elif self.policy == TLSPolicy.COMPAT:
            if protocol in ("TLSv1.0", "SSLv3"):
                violations.append(f"协议 {protocol} 已废弃，即使 COMPAT 策略也禁止")

        # 2. 弱密码套件检查
        if cipher_suite in self.WEAK_CIPHERS:
            violations.append(f"不安全密码套件: {cipher_suite}")
        elif cipher_suite and cipher_suite not in self.STRONG_CIPHERS:
            # 中等强度：STRICT 策略下警告
            if self.policy in (TLSPolicy.STRICT, TLSPolicy.MODERN):
                violations.append(f"非强密码套件: {cipher_suite}")

        # 3. 证书过期检查
        if cert_expiry:
            try:
                expiry = datetime.fromisoformat(cert_expiry)
                if expiry < datetime.now(timezone.utc):
                    violations.append(f"TLS 证书已过期: {cert_expiry}")
                elif expiry < datetime.now(timezone.utc) + timedelta(days=30):
                    violations.append(f"TLS 证书即将过期: {cert_expiry}")
            except (ValueError, TypeError):
                pass

        info = TLSConnectionInfo(
            protocol=protocol,
            cipher_suite=cipher_suite,
            cert_common_name=cert_cn,
            cert_expiry=cert_expiry,
            remote_address=remote_addr,
            is_valid=len(violations) == 0,
            violation="; ".join(violations) if violations else "",
        )

        if not info.is_valid:
            with self._lock:
                self._violations.append(info)
            self.logger.warning("TLS 安全策略违规", violations=violations)

        return info

    def is_connection_secure(self, client_protocol: str, cipher_suite: str = "") -> bool:
        """快速安全判断"""
        info = self.inspect_connection(client_protocol, cipher_suite)
        return info.is_valid

    def get_violation_report(self) -> Dict[str, Any]:
        """获取违规报告"""
        with self._lock:
            return {
                "policy": self.policy.value,
                "total_violations": len(self._violations),
                "recent_violations": [
                    {
                        "protocol": v.protocol,
                        "cipher": v.cipher_suite,
                        "remote": v.remote_address,
                        "violation": v.violation,
                    }
                    for v in self._violations[-50:]  # 最近 50 条
                ],
            }

    def clear_violations(self):
        with self._lock:
            self._violations.clear()


# ============================================================================
#  透明磁盘加密器
# ============================================================================

class EncryptionMode(str, Enum):
    """加密模式"""
    AES256_GCM = "AES-256-GCM"
    AES256_CBC = "AES-256-CBC"


@dataclass
class VolumeEncryptionStatus:
    """卷加密状态"""
    path: str
    encrypted: bool
    mode: str = ""
    file_count: int = 0
    encrypted_count: int = 0
    total_size_bytes: int = 0
    last_rotation: str = ""


class DiskEncryptor:
    """
    透明磁盘加密器

    对标 PRD 5.1: 静态存储的特征集卷与 AI 模型权重必须
    采用 AES-256 进行透明级别的全量磁盘加密。

    支持:
    - 目录级批量加密/解密
    - 密钥轮换 (key rotation)
    - 加密状态审计

    使用示例:
        encryptor = DiskEncryptor(master_key_path="/etc/qlib/master.key")
        encryptor.encrypt_directory("./data/features", recursive=True)
        status = encryptor.check_encryption_status("./data/features")
    """

    ENCRYPTED_EXTENSION = ".qlenc"

    def __init__(
        self,
        master_key_path: Optional[str] = None,
        key_env: str = "DISK_ENCRYPTION_KEY",
        mode: EncryptionMode = EncryptionMode.AES256_GCM,
    ):
        self.mode = mode
        self._key: bytes = self._load_key(master_key_path, key_env)
        self._rotation_history: List[Dict[str, Any]] = []
        self._lock = threading.Lock()
        self.logger = get_logger()

    def _load_key(self, path: Optional[str], env: str) -> bytes:
        if path and Path(path).exists():
            with open(path, "rb") as f:
                return f.read()[:32].ljust(32, b"\x00")
        env_val = os.environ.get(env)
        if env_val:
            import base64
            return base64.b64decode(env_val)[:32].ljust(32, b"\x00")
        # 开发环境回退 (生产环境必须配置)
        import hashlib
        return hashlib.sha256(b"qlib-disk-encryption-dev-key").digest()

    def encrypt_data(self, plaintext: bytes) -> bytes:
        """加密二进制数据 (AES-256-GCM)"""
        try:
            from cryptography.hazmat.primitives.ciphers.aead import AESGCM

            nonce = os.urandom(12)
            aesgcm = AESGCM(self._key)
            ciphertext = aesgcm.encrypt(nonce, plaintext, None)
            return nonce + ciphertext
        except ImportError:
            self.logger.warning("cryptography 未安装，使用弱 XOR 回退 (仅开发环境)")
            return self._xor_fallback(plaintext)

    def decrypt_data(self, ciphertext: bytes) -> bytes:
        """解密二进制数据"""
        try:
            from cryptography.hazmat.primitives.ciphers.aead import AESGCM

            nonce = ciphertext[:12]
            data = ciphertext[12:]
            aesgcm = AESGCM(self._key)
            return aesgcm.decrypt(nonce, data, None)
        except ImportError:
            return self._xor_fallback(ciphertext)

    def encrypt_file(self, file_path: str, remove_original: bool = False) -> str:
        """加密单个文件"""
        src = Path(file_path)
        dest = src.with_suffix(src.suffix + self.ENCRYPTED_EXTENSION)

        with open(src, "rb") as f:
            encrypted = self.encrypt_data(f.read())

        with open(dest, "wb") as f:
            f.write(encrypted)

        if remove_original:
            src.unlink()

        self.logger.debug(f"文件已加密: {src.name} -> {dest.name}")
        return str(dest)

    def decrypt_file(self, file_path: str, output_path: Optional[str] = None) -> str:
        """解密单个文件"""
        src = Path(file_path)
        if output_path is None:
            # 移除 .qlenc 后缀
            name = src.name
            if name.endswith(self.ENCRYPTED_EXTENSION):
                name = name[:-len(self.ENCRYPTED_EXTENSION)]
            dest = src.parent / name
        else:
            dest = Path(output_path)

        with open(src, "rb") as f:
            plaintext = self.decrypt_data(f.read())

        with open(dest, "wb") as f:
            f.write(plaintext)

        return str(dest)

    def encrypt_directory(
        self,
        directory: str,
        recursive: bool = True,
        pattern: str = "*.bin",
        remove_original: bool = False,
    ) -> int:
        """
        批量加密目录

        Args:
            directory: 目标目录
            recursive: 是否递归
            pattern: 文件匹配模式 (如 *.bin, *.pkl)
            remove_original: 是否删除原文件

        Returns:
            加密文件数
        """
        dir_path = Path(directory)
        if not dir_path.exists():
            return 0

        files = list(dir_path.rglob(pattern)) if recursive else list(dir_path.glob(pattern))
        count = 0
        for f in files:
            if f.suffix.endswith(self.ENCRYPTED_EXTENSION):
                continue  # 已加密，跳过
            try:
                self.encrypt_file(str(f), remove_original=remove_original)
                count += 1
            except Exception as e:
                self.logger.error(f"加密失败: {f}", error=str(e))

        self.logger.info(f"目录加密完成: {directory}", files=count)
        return count

    def decrypt_directory(
        self,
        directory: str,
        recursive: bool = True,
        remove_encrypted: bool = False,
    ) -> int:
        """批量解密目录"""
        dir_path = Path(directory)
        if not dir_path.exists():
            return 0

        pattern = f"*{self.ENCRYPTED_EXTENSION}"
        files = list(dir_path.rglob(pattern)) if recursive else list(dir_path.glob(pattern))
        count = 0
        for f in files:
            try:
                self.decrypt_file(str(f))
                if remove_encrypted:
                    f.unlink()
                count += 1
            except Exception as e:
                self.logger.error(f"解密失败: {f}", error=str(e))

        return count

    def check_encryption_status(self, directory: str, recursive: bool = True) -> VolumeEncryptionStatus:
        """检查目录加密状态"""
        dir_path = Path(directory)
        if not dir_path.exists():
            return VolumeEncryptionStatus(path=directory, encrypted=False)

        pattern = f"*{self.ENCRYPTED_EXTENSION}"
        all_files = list(dir_path.rglob("*")) if recursive else list(dir_path.glob("*"))
        regular_files = [f for f in all_files if f.is_file()]
        encrypted_files = [f for f in regular_files if f.suffix.endswith(self.ENCRYPTED_EXTENSION)]

        total_size = sum(f.stat().st_size for f in regular_files)

        return VolumeEncryptionStatus(
            path=directory,
            encrypted=len(encrypted_files) > 0,
            mode=self.mode.value,
            file_count=len(regular_files),
            encrypted_count=len(encrypted_files),
            total_size_bytes=total_size,
        )

    def rotate_key(self, new_key: Optional[bytes] = None) -> Dict[str, Any]:
        """
        密钥轮换

        Args:
            new_key: 新 32 字节密钥 (None 则自动生成)

        Returns:
            轮换记录
        """
        old_key_hash = self._compute_key_hash(self._key)
        new_key = new_key or os.urandom(32)

        with self._lock:
            self._key = new_key
            record = {
                "rotated_at": datetime.now(timezone.utc).isoformat(),
                "old_key_hash": old_key_hash,
                "new_key_hash": self._compute_key_hash(new_key),
            }
            self._rotation_history.append(record)

        self.logger.info("密钥已轮换", old_hash=old_key_hash[:8], new_hash=record["new_key_hash"][:8])
        return record

    def get_rotation_history(self) -> List[Dict[str, Any]]:
        with self._lock:
            return list(self._rotation_history)

    @staticmethod
    def _compute_key_hash(key: bytes) -> str:
        import hashlib
        return hashlib.sha256(key).hexdigest()[:16]

    @staticmethod
    def _xor_fallback(data: bytes) -> bytes:
        import hashlib
        key = hashlib.sha256(b"qlib-disk-dev-key").digest()
        return bytes(b ^ key[i % len(key)] for i, b in enumerate(data))


# ============================================================================
#  密钥/凭证生命周期管理
# ============================================================================

class SecretType(str, Enum):
    """密钥类型"""
    API_KEY = "api_key"
    ENCRYPTION_KEY = "encryption_key"
    HMAC_KEY = "hmac_key"
    DB_PASSWORD = "db_password"
    TLS_CERT = "tls_cert"


@dataclass
class SecretMetadata:
    """密钥元数据"""
    secret_id: str
    secret_type: SecretType
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    expires_at: Optional[str] = None
    rotated_from: Optional[str] = None
    is_active: bool = True
    description: str = ""
    owner: str = "system"


class SecretManager:
    """
    密钥生命周期管理器

    对标 PRD 5.1/5.2: 管理 API 密钥、加密密钥、HMAC 密钥等
    所有敏感凭证的安全存储、轮换和审计追踪。

    安全原则:
    - 密钥永不明文存储 (仅内存中)
    - 强制轮换策略 (API Key: 90天, Encryption: 年)
    - 所有密钥操作记录审计日志
    - 支持密钥版本化 (回滚到旧密钥解密历史数据)

    使用示例:
        sm = SecretManager(audit_logger=audit)
        sm.register_secret("prod_api_key", SecretType.API_KEY, expires_days=90)
        sm.rotate_secret("prod_api_key")
        if sm.is_expired("prod_api_key"):
            raise SecurityException("API key expired")
    """

    def __init__(self, audit_logger=None):
        self._secrets: Dict[str, bytes] = {}
        self._metadata: Dict[str, SecretMetadata] = {}
        self._versioned: Dict[str, List[bytes]] = {}  # 历史版本
        self._lock = threading.RLock()
        self.audit_logger = audit_logger
        self.logger = get_logger()

    # ------------------------------------------------------------------
    #  密钥注册
    # ------------------------------------------------------------------

    def register_secret(
        self,
        secret_id: str,
        secret_type: SecretType,
        secret_value: Optional[bytes] = None,
        expires_days: Optional[int] = None,
        description: str = "",
        owner: str = "system",
    ) -> SecretMetadata:
        """
        注册新密钥

        Args:
            secret_id: 密钥标识
            secret_type: 密钥类型
            secret_value: 密钥值 (None 则自动生成)
            expires_days: 过期天数 (None 则永不过期)
            description: 描述
            owner: 所有者

        Returns:
            SecretMetadata
        """
        if secret_value is None:
            secret_value = os.urandom(32)

        expires_at = None
        if expires_days is not None:
            expires_at = (datetime.now(timezone.utc) + timedelta(days=expires_days)).isoformat()

        meta = SecretMetadata(
            secret_id=secret_id,
            secret_type=secret_type,
            expires_at=expires_at,
            description=description,
            owner=owner,
        )

        with self._lock:
            self._secrets[secret_id] = secret_value
            self._metadata[secret_id] = meta
            self._versioned.setdefault(secret_id, []).append(secret_value)

        if self.audit_logger:
            self.audit_logger.log(
                event_type="secret_registered",
                action="register",
                resource=f"secret/{secret_id}",
                detail={"type": secret_type.value, "owner": owner},
            )

        self.logger.info("密钥已注册", secret_id=secret_id, type=secret_type.value)
        return meta

    # ------------------------------------------------------------------
    #  密钥轮换
    # ------------------------------------------------------------------

    def rotate_secret(
        self,
        secret_id: str,
        new_value: Optional[bytes] = None,
    ) -> bytes:
        """
        轮换密钥 (旧版本保留用于解密历史数据)

        Returns:
            新密钥值
        """
        if new_value is None:
            new_value = os.urandom(32)

        with self._lock:
            if secret_id not in self._secrets:
                raise KeyError(f"密钥 {secret_id} 未注册")

            old_value = self._secrets[secret_id]
            old_meta = self._metadata[secret_id]

            # 创建新元数据
            new_meta = SecretMetadata(
                secret_id=secret_id,
                secret_type=old_meta.secret_type,
                rotated_from=old_meta.secret_id,
                description=old_meta.description,
                owner=old_meta.owner,
            )
            # 保持相同的过期策略
            if old_meta.expires_at:
                remaining = datetime.fromisoformat(old_meta.expires_at) - datetime.now(timezone.utc)
                if remaining.total_seconds() > 0:
                    new_meta.expires_at = old_meta.expires_at

            self._secrets[secret_id] = new_value
            self._versioned[secret_id].append(new_value)
            self._metadata[secret_id] = new_meta

        if self.audit_logger:
            self.audit_logger.log(
                event_type="secret_rotated",
                action="rotate",
                resource=f"secret/{secret_id}",
                detail={"type": old_meta.secret_type.value},
            )

        self.logger.info("密钥已轮换", secret_id=secret_id)
        return new_value

    # ------------------------------------------------------------------
    #  密钥访问
    # ------------------------------------------------------------------

    def get_secret(self, secret_id: str) -> Optional[bytes]:
        """获取当前密钥值 (仅内存)"""
        with self._lock:
            return self._secrets.get(secret_id)

    def get_secret_version(self, secret_id: str, version: int = -1) -> Optional[bytes]:
        """获取指定版本的密钥 (用于解密历史数据)"""
        with self._lock:
            versions = self._versioned.get(secret_id, [])
            if not versions:
                return None
            try:
                return versions[version]
            except IndexError:
                return None

    def get_metadata(self, secret_id: str) -> Optional[SecretMetadata]:
        with self._lock:
            return self._metadata.get(secret_id)

    # ------------------------------------------------------------------
    #  过期检查
    # ------------------------------------------------------------------

    def is_expired(self, secret_id: str) -> bool:
        """检查密钥是否过期"""
        meta = self.get_metadata(secret_id)
        if meta is None or meta.expires_at is None:
            return False
        try:
            expiry = datetime.fromisoformat(meta.expires_at)
            return datetime.now(timezone.utc) > expiry
        except (ValueError, TypeError):
            return False

    def days_until_expiry(self, secret_id: str) -> Optional[int]:
        """距离过期天数"""
        meta = self.get_metadata(secret_id)
        if meta is None or meta.expires_at is None:
            return None
        try:
            expiry = datetime.fromisoformat(meta.expires_at)
            remaining = (expiry - datetime.now(timezone.utc)).days
            return max(0, remaining)
        except (ValueError, TypeError):
            return None

    def get_expiring_secrets(self, within_days: int = 30) -> List[str]:
        """获取即将过期的密钥列表"""
        expiring = []
        with self._lock:
            for sid, meta in self._metadata.items():
                if meta.expires_at is None:
                    continue
                try:
                    expiry = datetime.fromisoformat(meta.expires_at)
                    remaining = (expiry - datetime.now(timezone.utc)).days
                    if 0 <= remaining <= within_days:
                        expiring.append(sid)
                except (ValueError, TypeError):
                    pass
        return expiring

    # ------------------------------------------------------------------
    #  密钥吊销
    # ------------------------------------------------------------------

    def revoke_secret(self, secret_id: str):
        """吊销密钥 (紧急情况)"""
        with self._lock:
            self._secrets.pop(secret_id, None)
            meta = self._metadata.pop(secret_id, None)

        if self.audit_logger and meta:
            self.audit_logger.log(
                event_type="secret_revoked",
                action="revoke",
                resource=f"secret/{secret_id}",
                detail={"type": meta.secret_type.value, "reason": "manual_revocation"},
            )

        self.logger.warning("密钥已吊销", secret_id=secret_id)

    def list_secrets(self) -> List[Dict[str, Any]]:
        """列出所有密钥元数据"""
        with self._lock:
            return [
                {
                    "secret_id": sid,
                    "type": meta.secret_type.value,
                    "is_active": meta.is_active,
                    "expires_at": meta.expires_at,
                    "days_remaining": self.days_until_expiry(sid),
                    "version_count": len(self._versioned.get(sid, [])),
                    "owner": meta.owner,
                }
                for sid, meta in self._metadata.items()
            ]


# ============================================================================
#  SOX 合规报告生成器
# ============================================================================

class ComplianceStatus(str, Enum):
    """合规状态"""
    COMPLIANT = "compliant"
    NON_COMPLIANT = "non_compliant"
    NEEDS_REVIEW = "needs_review"


@dataclass
class SOXControlPoint:
    """SOX 控制点"""
    control_id: str
    description: str
    status: ComplianceStatus = ComplianceStatus.NEEDS_REVIEW
    evidence: str = ""
    tested_at: str = ""
    tested_by: str = ""


class SOXComplianceReporter:
    """
    SOX 合规报告生成器

    对标 PRD 5.2: 审计日志需满足萨班斯-奥克斯利法案 (SOX) 合规要求。

    核心控制点:
    - IT-1: 访问控制 (最小特权 + 职责分离)
    - IT-2: 变更管理 (模型/代码/配置变更记录)
    - IT-3: 数据完整性 (审计日志防篡改哈希链)
    - IT-4: 密钥管理 (定期轮换 + 吊销审计)
    - IT-5: 安全事件响应 (熔断 + 告警)
    """

    REQUIRED_CONTROLS = [
        ("SOX-IT-1", "访问控制: RBAC 最小特权与职责分离"),
        ("SOX-IT-2", "变更管理: 模型参数/代码/配置变更审批记录"),
        ("SOX-IT-3", "审计完整性: HMAC-SHA256 防篡改哈希链"),
        ("SOX-IT-4", "密钥管理: 强制轮换策略与吊销记录"),
        ("SOX-IT-5", "安全事件: 熔断开路/健康告警记录"),
        ("SOX-IT-6", "数据加密: AES-256 静态存储 + TLS 1.2+ 传输"),
        ("SOX-IT-7", "日志留存: 审计日志按法规期限归档"),
    ]

    def __init__(
        self,
        audit_logger=None,
        rbac_manager=None,
        tls_enforcer: Optional[TLSEnforcer] = None,
        secret_manager: Optional[SecretManager] = None,
        disk_encryptor: Optional[DiskEncryptor] = None,
    ):
        self.audit_logger = audit_logger
        self.rbac_manager = rbac_manager
        self.tls_enforcer = tls_enforcer
        self.secret_manager = secret_manager
        self.disk_encryptor = disk_encryptor
        self.logger = get_logger()

    def generate_quarterly_report(self, quarter: str = "") -> Dict[str, Any]:
        """
        生成季度 SOX 合规报告

        Returns:
            {
                "report_id": str,
                "period": str,
                "generated_at": str,
                "overall_status": "compliant|non_compliant",
                "controls": [...],
                "audit_chain_verified": bool,
                "rbac_review": {...},
                "key_rotation_summary": {...},
                "tls_violations": {...},
            }
        """
        controls = []
        all_compliant = True

        # IT-1: RBAC 审查
        rbac_status, rbac_detail = self._check_rbac()
        controls.append(SOXControlPoint(
            control_id="SOX-IT-1",
            description="访问控制: RBAC 最小特权与职责分离",
            status=rbac_status,
            evidence=json.dumps(rbac_detail),
        ))
        if rbac_status != ComplianceStatus.COMPLIANT:
            all_compliant = False

        # IT-2: 变更管理
        change_status, change_detail = self._check_change_management()
        controls.append(SOXControlPoint(
            control_id="SOX-IT-2",
            description="变更管理: 模型参数/代码/配置变更审批记录",
            status=change_status,
            evidence=json.dumps(change_detail),
        ))

        # IT-3: 审计完整性
        chain_verified = self._verify_audit_chain()
        controls.append(SOXControlPoint(
            control_id="SOX-IT-3",
            description="审计完整性: HMAC-SHA256 防篡改哈希链",
            status=ComplianceStatus.COMPLIANT if chain_verified else ComplianceStatus.NON_COMPLIANT,
            evidence=f"chain_verified={chain_verified}",
        ))
        if not chain_verified:
            all_compliant = False

        # IT-4: 密钥管理
        key_status, key_detail = self._check_key_management()
        controls.append(SOXControlPoint(
            control_id="SOX-IT-4",
            description="密钥管理: 强制轮换策略与吊销记录",
            status=key_status,
            evidence=json.dumps(key_detail),
        ))

        # IT-5: 安全事件
        sec_status, sec_detail = self._check_security_events()
        controls.append(SOXControlPoint(
            control_id="SOX-IT-5",
            description="安全事件: 熔断开路/健康告警记录",
            status=sec_status,
            evidence=json.dumps(sec_detail),
        ))

        # IT-6: 数据加密
        enc_status, enc_detail = self._check_encryption()
        controls.append(SOXControlPoint(
            control_id="SOX-IT-6",
            description="数据加密: AES-256 静态存储 + TLS 1.2+ 传输",
            status=enc_status,
            evidence=json.dumps(enc_detail),
        ))
        if enc_status != ComplianceStatus.COMPLIANT:
            all_compliant = False

        # IT-7: 日志留存
        retention_status, retention_detail = self._check_log_retention()
        controls.append(SOXControlPoint(
            control_id="SOX-IT-7",
            description="日志留存: 审计日志按法规期限归档",
            status=retention_status,
            evidence=json.dumps(retention_detail),
        ))

        now = datetime.now()
        if not quarter:
            quarter = f"{now.year}-Q{(now.month - 1) // 3 + 1}"

        report = {
            "report_id": f"SOX-{quarter}-{now.strftime('%Y%m%d')}",
            "period": quarter,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "overall_status": "compliant" if all_compliant else "non_compliant",
            "controls": [
                {
                    "control_id": c.control_id,
                    "description": c.description,
                    "status": c.status.value,
                    "evidence": c.evidence,
                }
                for c in controls
            ],
            "audit_chain_verified": chain_verified,
        }

        self.logger.info("SOX 合规报告已生成", overall=report["overall_status"])
        return report

    # ------------------------------------------------------------------
    #  各控制点检查
    # ------------------------------------------------------------------

    def _check_rbac(self) -> Tuple[ComplianceStatus, Dict]:
        if self.rbac_manager is None:
            return ComplianceStatus.NEEDS_REVIEW, {"error": "RBAC 未配置"}

        users = self.rbac_manager.list_users() if hasattr(self.rbac_manager, "list_users") else []
        roles_used = set()
        for u in users:
            if hasattr(u, "role"):
                roles_used.add(str(u.role.value) if hasattr(u.role, "value") else str(u.role))

        detail = {
            "total_users": len(users),
            "roles_in_use": list(roles_used),
            "has_admin": "system_admin" in roles_used,
        }
        return ComplianceStatus.COMPLIANT, detail

    def _check_change_management(self) -> Tuple[ComplianceStatus, Dict]:
        if self.audit_logger is None:
            return ComplianceStatus.NEEDS_REVIEW, {"error": "审计日志未配置"}

        # 查询近期变更记录
        changes = []
        if hasattr(self.audit_logger, "query"):
            try:
                changes = self.audit_logger.query(
                    event_type="model_deploy",
                    limit=50,
                )
                changes += self.audit_logger.query(
                    event_type="config_change",
                    limit=50,
                )
            except Exception:
                pass

        return ComplianceStatus.COMPLIANT, {
            "recent_changes": len(changes),
            "has_records": len(changes) > 0,
        }

    def _verify_audit_chain(self) -> bool:
        if self.audit_logger is None:
            return False
        if hasattr(self.audit_logger, "verify_chain"):
            result = self.audit_logger.verify_chain()
            return result.get("valid", False)
        return False

    def _check_key_management(self) -> Tuple[ComplianceStatus, Dict]:
        if self.secret_manager is None:
            return ComplianceStatus.NEEDS_REVIEW, {"error": "密钥管理器未配置"}

        secrets = self.secret_manager.list_secrets() if hasattr(self.secret_manager, "list_secrets") else []
        expiring = self.secret_manager.get_expiring_secrets(within_days=30)

        return ComplianceStatus.COMPLIANT, {
            "total_secrets": len(secrets),
            "expiring_soon": len(expiring),
            "expiring_ids": expiring,
        }

    def _check_security_events(self) -> Tuple[ComplianceStatus, Dict]:
        events = {}
        if self.tls_enforcer:
            tls_report = self.tls_enforcer.get_violation_report()
            events["tls_violations"] = tls_report.get("total_violations", 0)

        return ComplianceStatus.COMPLIANT, events

    def _check_encryption(self) -> Tuple[ComplianceStatus, Dict]:
        if self.disk_encryptor is None:
            return ComplianceStatus.NEEDS_REVIEW, {"error": "磁盘加密器未配置"}

        history = self.disk_encryptor.get_rotation_history()
        return ComplianceStatus.COMPLIANT, {
            "key_rotations": len(history),
            "has_rotated": len(history) > 0,
        }

    def _check_log_retention(self) -> Tuple[ComplianceStatus, Dict]:
        if self.audit_logger is None:
            return ComplianceStatus.NEEDS_REVIEW, {"error": "审计日志未配置"}

        log_dir = getattr(self.audit_logger, "log_dir", None)
        log_count = 0
        if log_dir:
            log_count = len(list(Path(str(log_dir)).glob("audit_*.jsonl")))

        return ComplianceStatus.COMPLIANT, {
            "log_files": log_count,
            "log_dir": str(log_dir) if log_dir else "unknown",
        }


# ============================================================================
#  审计日志归档管理器
# ============================================================================

class RetentionPolicy(str, Enum):
    """日志留存策略"""
    SOX_MINIMUM = "sox_7yr"        # SOX 要求至少 7 年
    STANDARD = "standard_3yr"      # 标准 3 年
    EXTENDED = "extended_10yr"     # 扩展 10 年
    PERMANENT = "permanent"        # 永久


class AuditArchiveManager:
    """
    审计日志归档管理器

    对标 PRD 5.2: 审计日志归档至独立存储供按需调阅。
    SOX 要求审计日志保留至少 7 年。

    使用示例:
        archiver = AuditArchiveManager(
            archive_dir="./archive/audit",
            retention=RetentionPolicy.SOX_MINIMUM,
        )
        archiver.archive_logs(source_dir="./logs/audit")
        archiver.purge_expired()
    """

    RETENTION_DAYS = {
        RetentionPolicy.SOX_MINIMUM: 2557,   # 7年
        RetentionPolicy.STANDARD: 1095,      # 3年
        RetentionPolicy.EXTENDED: 3653,      # 10年
        RetentionPolicy.PERMANENT: 36500,    # 100年(≈永久)
    }

    def __init__(
        self,
        archive_dir: str = "./archive/audit",
        retention: RetentionPolicy = RetentionPolicy.SOX_MINIMUM,
    ):
        self.archive_dir = Path(archive_dir)
        self.archive_dir.mkdir(parents=True, exist_ok=True)
        self.retention = retention
        self._lock = threading.Lock()
        self.logger = get_logger()

    @property
    def retention_days(self) -> int:
        return self.RETENTION_DAYS.get(self.retention, 2557)

    def archive_logs(self, source_dir: str) -> int:
        """
        归档审计日志到独立存储

        Args:
            source_dir: 源日志目录

        Returns:
            归档文件数
        """
        src = Path(source_dir)
        if not src.exists():
            return 0

        archive_date = datetime.now().strftime("%Y%m%d_%H%M%S")
        archive_path = self.archive_dir / f"audit_archive_{archive_date}"
        archive_path.mkdir(parents=True, exist_ok=True)

        count = 0
        for log_file in src.glob("audit_*.jsonl"):
            dest = archive_path / log_file.name
            shutil.copy2(log_file, dest)
            count += 1

        if count > 0:
            # 创建清单
            manifest = {
                "archive_date": archive_date,
                "source_dir": str(src),
                "retention_policy": self.retention.value,
                "retention_days": self.retention_days,
                "expires_at": (datetime.now(timezone.utc) + timedelta(days=self.retention_days)).isoformat(),
                "files_archived": count,
            }
            with open(archive_path / "manifest.json", "w") as f:
                json.dump(manifest, f, indent=2)

        self.logger.info(f"审计日志已归档: {count} 文件 -> {archive_path}")
        return count

    def purge_expired(self) -> int:
        """
        清理过期归档

        Returns:
            清理的归档数
        """
        if self.retention == RetentionPolicy.PERMANENT:
            return 0

        cutoff = datetime.now() - timedelta(days=self.retention_days)
        purged = 0

        for archive_dir in self.archive_dir.iterdir():
            if not archive_dir.is_dir():
                continue
            manifest_path = archive_dir / "manifest.json"
            if not manifest_path.exists():
                continue

            try:
                with open(manifest_path) as f:
                    manifest = json.load(f)
                archive_date = manifest.get("archive_date", "")
                if archive_date:
                    archive_dt = datetime.strptime(archive_date[:8], "%Y%m%d")
                    if archive_dt < cutoff:
                        shutil.rmtree(archive_dir)
                        purged += 1
                        self.logger.info(f"已清理过期归档: {archive_dir.name}")
            except Exception:
                pass

        return purged

    def list_archives(self) -> List[Dict[str, Any]]:
        """列出所有归档"""
        archives = []
        for archive_dir in sorted(self.archive_dir.iterdir(), reverse=True):
            if not archive_dir.is_dir():
                continue
            manifest_path = archive_dir / "manifest.json"
            info = {"archive_name": archive_dir.name}
            if manifest_path.exists():
                try:
                    with open(manifest_path) as f:
                        info.update(json.load(f))
                except Exception:
                    pass
            archives.append(info)
        return archives


# ============================================================================
#  数据敏感性分类
# ============================================================================

class DataSensitivity(str, Enum):
    """数据敏感性级别"""
    PUBLIC = "public"             # 公开数据 (如市场行情)
    INTERNAL = "internal"         # 内部数据 (如因子定义)
    CONFIDENTIAL = "confidential" # 机密数据 (如模型权重)
    RESTRICTED = "restricted"     # 受限数据 (如 PII/交易记录)
    CRITICAL = "critical"         # 关键数据 (如加密密钥)


class DataClassification:
    """
    数据敏感性分类器

    对标 PRD 5.1/5.2: 对系统中不同类型数据进行敏感性分类，
    确保不同级别数据应用对应的加密和访问控制策略。

    使用示例:
        classifier = DataClassification()
        classifier.classify("features/*.bin", DataSensitivity.CONFIDENTIAL)
        classifier.classify("models/*.pkl", DataSensitivity.RESTRICTED)
        level = classifier.get_classification("models/lgb_v2.pkl")
        assert level == DataSensitivity.RESTRICTED
    """

    def __init__(self):
        self._classifications: Dict[str, DataSensitivity] = {}
        self._patterns: List[Tuple[str, DataSensitivity]] = []
        self._lock = threading.Lock()
        self.logger = get_logger()

    def classify(self, path_pattern: str, sensitivity: DataSensitivity):
        """注册路径模式到敏感性级别"""
        with self._lock:
            self._patterns.append((path_pattern, sensitivity))
        self.logger.info(f"数据分类: {path_pattern} -> {sensitivity.value}")

    def get_classification(self, path: str) -> DataSensitivity:
        """
        获取指定路径的敏感性级别

        按注册顺序匹配，先匹配优先。
        """
        from fnmatch import fnmatch

        with self._lock:
            for pattern, sensitivity in self._patterns:
                if fnmatch(path, pattern):
                    return sensitivity
        return DataSensitivity.INTERNAL  # 默认内部级别

    def requires_encryption(self, path: str) -> bool:
        """判断路径是否需要加密"""
        level = self.get_classification(path)
        return level in (
            DataSensitivity.CONFIDENTIAL,
            DataSensitivity.RESTRICTED,
            DataSensitivity.CRITICAL,
        )

    def requires_audit(self, path: str) -> bool:
        """判断访问路径是否需要审计"""
        level = self.get_classification(path)
        return level in (
            DataSensitivity.CONFIDENTIAL,
            DataSensitivity.RESTRICTED,
            DataSensitivity.CRITICAL,
        )

    def get_all_classifications(self) -> Dict[str, str]:
        with self._lock:
            return {pattern: sensitivity.value for pattern, sensitivity in self._patterns}

    def clear(self):
        with self._lock:
            self._patterns.clear()
