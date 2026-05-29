"""
金融级安全模块 (Security Module)

AES-256-GCM 加密 + TLS 验证 + 防篡改审计日志 + RBAC 访问控制
+ SOX 合规报告 + 透明磁盘加密 + 密钥生命周期管理
"""

from src.security.security import (
    AES256Encryptor,
    TLSValidator,
    AuditLogger,
    AuditEntry,
    RBACManager,
    Role,
    User,
    PERMISSIONS,
    FineGrainedPerm,
    FINEGRAINED_PERMISSIONS,
    PERMISSION_DEPENDENCIES,
    AccessGrant,
    AccessRequest,
)

from src.security.compliance import (
    DiskEncryptor,
    TLSEnforcer,
    TLSVersion,
    TLSPolicy,
    TLSConnectionInfo,
    SOXComplianceReporter,
    SOXControlPoint,
    AuditArchiveManager,
    RetentionPolicy,
    SecretManager,
    SecretType,
    SecretMetadata,
    DataClassification,
    DataSensitivity,
    EncryptionMode,
    VolumeEncryptionStatus,
)

from src.security.pm_gate import (
    PMGateController,
    GateState,
    GateDimension,
    GateAction,
    GateClosedError,
)

__all__ = [
    # 核心安全
    "AES256Encryptor",
    "TLSValidator",
    "AuditLogger",
    "AuditEntry",
    "RBACManager",
    "Role",
    "User",
    "PERMISSIONS",
    # 细粒度权限
    "FineGrainedPerm",
    "FINEGRAINED_PERMISSIONS",
    "PERMISSION_DEPENDENCIES",
    "AccessGrant",
    "AccessRequest",
    # 合规
    "DiskEncryptor",
    "TLSEnforcer",
    "TLSVersion",
    "TLSPolicy",
    "TLSConnectionInfo",
    "SOXComplianceReporter",
    "SOXControlPoint",
    "AuditArchiveManager",
    "RetentionPolicy",
    "SecretManager",
    "SecretType",
    "SecretMetadata",
    "DataClassification",
    "DataSensitivity",
    "EncryptionMode",
    "VolumeEncryptionStatus",
    # PM 熔断门控
    "PMGateController",
    "GateState",
    "GateDimension",
    "GateAction",
    "GateClosedError",
]