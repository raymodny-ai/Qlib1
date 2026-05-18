"""
安全模块单元测试 — AES256Encryptor / AuditLogger / RBACManager
细粒度权限矩阵 / 时间限权 / 审批工作流
"""

import os
import json
import tempfile
import pytest
from src.security.security import (
    AES256Encryptor,
    TLSValidator,
    AuditLogger,
    AuditEntry,
    RBACManager,
    Role,
    User,
    FineGrainedPerm,
    AccessGrant,
    AccessRequest,
    FINEGRAINED_PERMISSIONS,
    PERMISSION_DEPENDENCIES,
)


# ==========================================================================
#  模块级 Fixtures (跨类共享)
# ==========================================================================

@pytest.fixture
def rbac_with_audit():
    """模块级 fixture: RBAC 管理器 + 审计日志"""
    with tempfile.TemporaryDirectory() as tmp:
        audit = AuditLogger(log_dir=tmp)
        mgr = RBACManager(audit_logger=audit)
        mgr.add_user(User("qr_001", "Alice", Role.QUANT_RESEARCHER))
        mgr.add_user(User("pm_001", "Bob", Role.PORTFOLIO_MANAGER))
        mgr.add_user(User("sa_001", "Eve", Role.SYSTEM_ADMIN))
        yield mgr, audit


class TestAES256Encryptor:
    """AES-256 加密器测试"""

    def test_key_validation(self):
        with pytest.raises(ValueError):
            AES256Encryptor(key=b"short")

    def test_encrypt_decrypt_bytes(self):
        aes = AES256Encryptor()
        plaintext = b"Hello, Qlib Security!"
        ciphertext = aes.encrypt(plaintext)
        assert ciphertext != plaintext
        decrypted = aes.decrypt(ciphertext)
        assert decrypted == plaintext

    def test_encrypt_decrypt_string(self):
        aes = AES256Encryptor()
        plaintext = "美股基本面分析系统"
        ciphertext = aes.encrypt(plaintext)
        decrypted = aes.decrypt(ciphertext)
        assert decrypted.decode("utf-8") == plaintext

    def test_encrypt_produces_different_ciphertext(self):
        aes = AES256Encryptor()
        c1 = aes.encrypt(b"same data")
        c2 = aes.encrypt(b"same data")
        assert c1 != c2  # nonce 不同

    def test_file_encrypt_decrypt(self):
        aes = AES256Encryptor()
        with tempfile.TemporaryDirectory() as tmp:
            orig_path = os.path.join(tmp, "original.txt")
            with open(orig_path, "w") as f:
                f.write("sensitive financial data")

            enc_path = aes.encrypt_file(orig_path)
            assert os.path.exists(enc_path)

            dec_path = aes.decrypt_file(enc_path)
            with open(dec_path, "r") as f:
                assert f.read() == "sensitive financial data"

    def test_generate_key(self):
        key = AES256Encryptor.generate_key()
        assert len(key) == 32

    def test_key_to_b64(self):
        key = b"a" * 32
        b64 = AES256Encryptor.key_to_b64(key)
        assert isinstance(b64, str)


class TestAuditLogger:
    """审计日志测试"""

    def test_log_entry(self):
        with tempfile.TemporaryDirectory() as tmp:
            audit = AuditLogger(log_dir=tmp)
            entry = audit.log(
                event_type="model_deploy",
                user="pm_zhang",
                role="portfolio_manager",
                action="deploy",
                resource="lgb_v2",
            )
            assert entry.event_type == "model_deploy"
            assert entry.hash_chain != ""

    def test_hash_chain_integrity(self):
        with tempfile.TemporaryDirectory() as tmp:
            audit = AuditLogger(log_dir=tmp)
            
            for i in range(5):
                audit.log("test_event", user=f"user_{i}")
            
            result = audit.verify_chain()
            assert result["valid"]
            assert len(result["violations"]) == 0

    def test_query_by_type(self):
        with tempfile.TemporaryDirectory() as tmp:
            audit = AuditLogger(log_dir=tmp)
            audit.log("login", user="alice")
            audit.log("logout", user="alice")
            audit.log("login", user="bob")
            
            logins = audit.query(event_type="login")
            assert len(logins) == 2

    def test_query_by_user(self):
        with tempfile.TemporaryDirectory() as tmp:
            audit = AuditLogger(log_dir=tmp)
            audit.log("login", user="alice")
            audit.log("login", user="bob")
            
            alice_events = audit.query(user="alice")
            assert len(alice_events) == 1

    def test_export_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            audit = AuditLogger(log_dir=tmp)
            audit.log("test_event", user="tester")
            
            report_path = os.path.join(tmp, "report.json")
            path = audit.export_report(report_path)
            assert os.path.exists(path)
            
            with open(path, "r") as f:
                report = json.load(f)
            assert report["total_events"] >= 1

    def test_audit_entry_to_dict(self):
        entry = AuditEntry(
            event_type="config_change",
            user="admin",
            action="update",
            resource="qlib_config.yaml",
        )
        d = entry.to_dict()
        assert d["event_type"] == "config_change"
        assert "hash_chain" in d


class TestRBACManager:
    """RBAC 访问控制测试"""

    @pytest.fixture
    def rbac(self):
        mgr = RBACManager()
        mgr.add_user(User("qr_001", "Alice", Role.QUANT_RESEARCHER))
        mgr.add_user(User("pm_001", "Bob", Role.PORTFOLIO_MANAGER))
        mgr.add_user(User("da_001", "Charlie", Role.DATA_ADMIN))
        mgr.add_user(User("ca_001", "Diana", Role.COMPLIANCE_AUDITOR))
        mgr.add_user(User("sa_001", "Eve", Role.SYSTEM_ADMIN))
        return mgr

    # rbac_with_audit fixture is module-level (see below)

    def test_researcher_can_train(self, rbac):
        assert rbac.can_train_model("qr_001")

    def test_researcher_cannot_push_signal(self, rbac):
        assert not rbac.can_push_signal("qr_001")

    def test_pm_can_push_signal(self, rbac):
        assert rbac.can_push_signal("pm_001")

    def test_pm_can_emergency_stop(self, rbac):
        assert rbac.can_emergency_stop("pm_001")

    def test_pm_inherits_researcher_read(self, rbac):
        # PM 继承 Researcher 的只读权限
        assert rbac.check_permission("pm_001", "data:read")
        assert rbac.check_permission("pm_001", "report:read")

    def test_compliance_auditor_read_only(self, rbac):
        assert rbac.check_permission("ca_001", "audit:read")
        assert rbac.check_permission("ca_001", "compliance:export")
        assert not rbac.check_permission("ca_001", "model:train")
        assert not rbac.can_push_signal("ca_001")

    def test_admin_has_all_permissions(self, rbac):
        assert rbac.can_train_model("sa_001")
        assert rbac.can_push_signal("sa_001")
        assert rbac.can_emergency_stop("sa_001")
        assert rbac.check_permission("sa_001", "data:write")

    def test_unknown_user(self, rbac):
        assert not rbac.check_permission("unknown", "data:read")

    def test_inactive_user(self, rbac):
        # 添加未激活用户
        rbac.add_user(User("inactive", "Z", Role.QUANT_RESEARCHER, active=False))
        assert not rbac.check_permission("inactive", "model:train")

    def test_assert_permission_raises(self, rbac):
        with pytest.raises(PermissionError):
            rbac.assert_permission("qr_001", "signal:approve")

    def test_assert_permission_passes(self, rbac):
        rbac.assert_permission("pm_001", "signal:approve")

    def test_list_users(self, rbac):
        users = rbac.list_users()
        assert len(users) == 5

    def test_get_all_roles(self):
        roles = RBACManager.get_all_roles()
        assert len(roles) == 5  # 5 种角色

    def test_remove_user(self, rbac):
        rbac.remove_user("qr_001")
        assert rbac.get_user("qr_001") is None

    def test_add_user(self, rbac):
        rbac.add_user(User("new", "New", Role.QUANT_RESEARCHER))
        assert rbac.get_user("new") is not None


# ==========================================================================
#  第6章: 细粒度权限矩阵测试
# ==========================================================================

class TestFineGrainedPermissions:
    """细粒度权限枚举与映射测试"""

    def test_finegrained_perm_values_unique(self):
        values = [p.value for p in FineGrainedPerm]
        assert len(values) == len(set(values))

    def test_all_roles_have_permissions(self):
        for role in Role:
            perms = FINEGRAINED_PERMISSIONS.get(role, set())
            assert len(perms) > 0, f"{role.value} has no finegrained permissions"

    def test_system_admin_has_wildcard(self):
        assert "*" in FINEGRAINED_PERMISSIONS[Role.SYSTEM_ADMIN]

    def test_quant_researcher_permissions(self):
        perms = FINEGRAINED_PERMISSIONS[Role.QUANT_RESEARCHER]
        assert FineGrainedPerm.FACTOR_WRITE_DEF in perms
        assert FineGrainedPerm.MODEL_TRAIN in perms
        # Researcher must NOT have signal approval
        assert FineGrainedPerm.SIGNAL_APPROVE not in perms

    def test_pm_has_emergency_stop(self):
        perms = FINEGRAINED_PERMISSIONS[Role.PORTFOLIO_MANAGER]
        assert FineGrainedPerm.SIGNAL_EMERGENCY_STOP in perms
        assert FineGrainedPerm.SIGNAL_APPROVE in perms
        assert FineGrainedPerm.RISK_CONFIGURE in perms

    def test_data_admin_permissions(self):
        perms = FINEGRAINED_PERMISSIONS[Role.DATA_ADMIN]
        assert FineGrainedPerm.DATA_DELETE in perms
        assert FineGrainedPerm.API_CONFIGURE in perms
        assert FineGrainedPerm.USER_CREATE in perms

    def test_compliance_auditor_readonly(self):
        perms = FINEGRAINED_PERMISSIONS[Role.COMPLIANCE_AUDITOR]
        assert FineGrainedPerm.AUDIT_READ in perms
        assert FineGrainedPerm.COMPLIANCE_EXPORT in perms
        # Auditor should NOT have write/data write
        assert FineGrainedPerm.DATA_WRITE not in perms
        assert FineGrainedPerm.MODEL_TRAIN not in perms

    def test_permission_dependencies_defined(self):
        assert "model:deploy" in PERMISSION_DEPENDENCIES
        assert "signal:approve" in PERMISSION_DEPENDENCIES
        assert "data:export" in PERMISSION_DEPENDENCIES

    def test_researcher_finegrained_check(self, rbac_with_audit):
        rbac, audit = rbac_with_audit
        rbac.add_user(User("qr", "Q", Role.QUANT_RESEARCHER))
        assert rbac.check_permission("qr", FineGrainedPerm.FACTOR_EXECUTE.value)
        assert rbac.check_permission("qr", FineGrainedPerm.MODEL_TRAIN.value)
        assert not rbac.check_permission("qr", FineGrainedPerm.SIGNAL_APPROVE.value)


# ==========================================================================
#  第6章: 时间限权测试
# ==========================================================================

class TestAccessGrant:
    """临时权限授权测试"""

    def test_grant_is_valid_when_active(self):
        grant = AccessGrant(
            user_id="qr_001",
            permission="signal:approve",
            granted_by="pm_001",
            reason="emergency",
        )
        assert grant.is_valid() is True  # No expiry = valid

    def test_grant_is_expired(self):
        grant = AccessGrant(
            user_id="qr_001",
            permission="signal:approve",
            granted_by="pm_001",
            expires_at="2020-01-01T00:00:00+00:00",
        )
        assert grant.is_expired() is True
        assert grant.is_valid() is False

    def test_grant_is_exhausted(self):
        grant = AccessGrant(
            user_id="qr_001",
            permission="signal:approve",
            granted_by="pm_001",
            max_uses=1,
            used_count=1,
        )
        assert grant.is_exhausted() is True

    def test_grant_unlimited_uses(self):
        grant = AccessGrant(
            user_id="qr_001",
            permission="signal:approve",
            granted_by="pm_001",
            max_uses=-1,
            used_count=999,
        )
        assert grant.is_exhausted() is False
        assert grant.is_valid() is True

    def test_grant_inactive(self):
        grant = AccessGrant(
            user_id="qr_001",
            permission="signal:approve",
            granted_by="pm_001",
            is_active=False,
        )
        assert grant.is_valid() is False


class TestTemporaryPermission:
    """临时权限集成测试"""

    def test_grant_temporary_permission(self, rbac_with_audit):
        rbac, audit = rbac_with_audit
        grant = rbac.grant_temporary_permission(
            user_id="qr_001",
            permission="signal:approve",
            granted_by="pm_001",
            duration_hours=24,
            reason="production deploy test",
        )
        assert grant.grant_id
        assert grant.user_id == "qr_001"
        assert grant.granted_by == "pm_001"

        # Researcher now has temporary signal approval
        assert rbac.check_permission("qr_001", "signal:approve")

    def test_temporary_grant_tracks_usage(self, rbac_with_audit):
        rbac, audit = rbac_with_audit
        rbac.grant_temporary_permission(
            user_id="qr_001",
            permission="signal:approve",
            granted_by="pm_001",
            max_uses=1,
        )
        # First use should work
        assert rbac.check_permission("qr_001", "signal:approve")
        # Second use should fail (exhausted)
        assert not rbac.check_permission("qr_001", "signal:approve")

    def test_revoke_temporary_grant(self, rbac_with_audit):
        rbac, audit = rbac_with_audit
        grant = rbac.grant_temporary_permission(
            user_id="qr_001",
            permission="signal:approve",
            granted_by="pm_001",
        )
        assert rbac.check_permission("qr_001", "signal:approve")

        rbac.revoke_temporary_permission(grant.grant_id, revoked_by="pm_001")
        assert not rbac.check_permission("qr_001", "signal:approve")

    def test_list_active_grants(self, rbac_with_audit):
        rbac, audit = rbac_with_audit
        rbac.grant_temporary_permission(
            user_id="qr_001",
            permission="signal:approve",
            granted_by="pm_001",
        )
        rbac.grant_temporary_permission(
            user_id="qr_001",
            permission="model:deploy",
            granted_by="pm_001",
            max_uses=-1,
        )
        active = rbac.list_active_grants(user_id="qr_001")
        assert len(active) == 2


# ==========================================================================
#  第6章: 审批工作流测试
# ==========================================================================

class TestAccessRequestWorkflow:
    """审批工作流集成测试"""

    def test_request_access(self, rbac_with_audit):
        rbac, audit = rbac_with_audit
        req = rbac.request_access(
            requester_id="qr_001",
            permission="signal:approve",
            reason="Need to push signal for production",
            duration_hours=8,
        )
        assert req.request_id
        assert req.status == "pending"
        assert req.requester_id == "qr_001"

        # Request should not yet grant permission
        assert not rbac.check_permission("qr_001", "signal:approve")

    def test_approve_request(self, rbac_with_audit):
        rbac, audit = rbac_with_audit
        req = rbac.request_access(
            requester_id="qr_001",
            permission="signal:approve",
            reason="urgent deploy",
            duration_hours=4,
        )
        grant = rbac.approve_request(
            request_id=req.request_id,
            approver_id="pm_001",
            resolution_note="Approved for emergency deployment",
        )
        assert grant is not None
        assert grant.permission == "signal:approve"

        # After approval, researcher has permission
        assert rbac.check_permission("qr_001", "signal:approve")

    def test_reject_request(self, rbac_with_audit):
        rbac, audit = rbac_with_audit
        req = rbac.request_access(
            requester_id="qr_001",
            permission="signal:approve",
            reason="test",
        )
        result = rbac.reject_request(
            request_id=req.request_id,
            approver_id="pm_001",
            resolution_note="Rejected - no justification",
        )
        assert result is True

        # After rejection, still no permission
        assert not rbac.check_permission("qr_001", "signal:approve")

    def test_list_pending_requests(self, rbac_with_audit):
        rbac, audit = rbac_with_audit
        rbac.request_access("qr_001", "signal:approve", reason="test1")
        rbac.request_access("qr_001", "model:deploy", reason="test2")

        pending = rbac.list_requests(status="pending")
        assert len(pending) == 2

    def test_approve_already_resolved_request(self, rbac_with_audit):
        rbac, audit = rbac_with_audit
        req = rbac.request_access("qr_001", "signal:approve", reason="test")
        rbac.reject_request(req.request_id, "pm_001")
        # Try to approve already-rejected request
        grant = rbac.approve_request(req.request_id, "pm_001")
        assert grant is None

    def test_reject_already_resolved_request(self, rbac_with_audit):
        rbac, audit = rbac_with_audit
        req = rbac.request_access("qr_001", "signal:approve", reason="test")
        rbac.approve_request(req.request_id, "pm_001")
        # Try to reject already-approved request
        result = rbac.reject_request(req.request_id, "pm_001")
        assert result is False

    def test_list_requests_by_requester(self, rbac_with_audit):
        rbac, audit = rbac_with_audit
        rbac.request_access("qr_001", "signal:approve", reason="req1")
        rbac.request_access("qr_001", "model:deploy", reason="req2")

        requests = rbac.list_requests(requester_id="qr_001")
        assert len(requests) == 2


# ==========================================================================
#  第6章: 权限依赖检查测试
# ==========================================================================

class TestPermissionDependencies:
    """权限依赖图测试"""

    @pytest.fixture
    def rbac(self):
        mgr = RBACManager()
        mgr.add_user(User("pm", "PM", Role.PORTFOLIO_MANAGER))
        mgr.add_user(User("admin", "Admin", Role.SYSTEM_ADMIN))
        return mgr

    def test_pm_can_deploy_model(self, rbac):
        # PM can read models but cannot train implicitly
        # deploy requires model:read + model:train
        # PM inherits from researcher which has model:train, so PM should be able to deploy
        result = rbac.check_permission("pm", "model:deploy")
        # PM inherits from Researcher (which has model:train)
        # But PM doesn't directly have model:train... wait, PM inherits from Researcher
        assert result is True  # PM inherits researcher permissions

    def test_admin_can_deploy_with_dependency(self, rbac):
        # Admin has *, so should pass all dependency checks
        assert rbac.check_permission("admin", "model:deploy")

    def test_signal_approve_requires_risk_read(self, rbac):
        # PM has both signal:read and risk:read
        assert rbac.check_permission("pm", "signal:approve")

    def test_get_finegrained_permissions(self, rbac):
        perms = rbac.get_finegrained_permissions(Role.QUANT_RESEARCHER)
        assert FineGrainedPerm.FACTOR_EXECUTE.value in perms
        assert FineGrainedPerm.MODEL_TRAIN.value in perms

    def test_get_all_permissions_combines_both(self, rbac):
        all_perms = rbac.get_all_permissions(Role.QUANT_RESEARCHER)
        # Should have both legacy and finegrained
        assert "model:train" in all_perms  # legacy
        assert FineGrainedPerm.FACTOR_EXECUTE.value in all_perms  # finegrained
