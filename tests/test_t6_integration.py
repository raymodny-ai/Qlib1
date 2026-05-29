"""
T6 集成验证: RBAC PM 熔断门控 + 合规审计端点

验证:
1. PMGateController 状态机与 RBAC 强制
2. 审计日志写入与哈希链完整性
3. API 端点模拟调用
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import tempfile
from pathlib import Path
from datetime import datetime, timezone


def test_pm_gate_basic():
    """测试 PMGateController 基础状态机"""
    print("=" * 60)
    print("TEST 1: PMGateController 基础状态机")
    print("=" * 60)

    from src.security.pm_gate import PMGateController, GateState, GateDimension, GateClosedError

    gate = PMGateController(min_action_interval=0)

    # 1. 初始状态
    assert gate.can_push_signal() == True, "signal gate should be open by default"
    assert gate.can_train_model() == True, "train gate should be open by default"
    assert gate.can_deploy_model() == False, "deploy gate should be closed by default"
    print("  ✓ 初始状态正确: signal=OPEN, train=OPEN, deploy=CLOSED")

    # 2. 无 RBAC 情况下, 操作门控 (开发环境)
    action = gate.emergency_stop(
        user_id="test_pm",
        dimension=GateDimension.SIGNAL,
        reason="测试熔断"
    )
    assert gate.can_push_signal() == False, "signal gate should be closed after stop"
    assert action.action == "emergency_stop"
    assert action.to_state == "closed"
    print("  ✓ 一键熔断成功: signal gate CLOSED")

    # 3. 重复熔断应报错
    try:
        gate.emergency_stop(user_id="test_pm", dimension=GateDimension.SIGNAL, reason="重复")
        assert False, "Should raise RuntimeError"
    except RuntimeError:
        print("  ✓ 重复熔断正确拒绝")

    # 4. 恢复放行
    action = gate.emergency_reopen(
        user_id="test_pm",
        dimension=GateDimension.SIGNAL,
        reason="测试恢复"
    )
    assert gate.can_push_signal() == True
    assert action.action == "emergency_reopen"
    assert action.to_state == "open"
    print("  ✓ 恢复放行成功: signal gate OPEN")

    # 5. GateClosedError
    try:
        raise GateClosedError("test")
    except GateClosedError as e:
        assert "test" in str(e)
        print("  ✓ GateClosedError 正常")

    # 6. 装饰器门控保护
    gate.emergency_stop(user_id="test_pm", dimension=GateDimension.SIGNAL, reason="装饰器测试")

    @gate.require_signal_gate_open
    def push_signal(data):
        return f"pushed: {data}"

    try:
        push_signal("test_signal")
        assert False, "Should raise GateClosedError"
    except GateClosedError:
        print("  ✓ require_signal_gate_open 装饰器正确拦截")

    gate.emergency_reopen(user_id="test_pm", dimension=GateDimension.SIGNAL, reason="恢复")
    result = push_signal("test_signal")
    assert result == "pushed: test_signal"
    print("  ✓ 门控恢复后装饰器正常放行")

    print("TEST 1 PASSED\n")


def test_pm_gate_rbac():
    """测试 PM Gate RBAC 强制"""
    print("=" * 60)
    print("TEST 2: PM Gate RBAC 强制验证")
    print("=" * 60)

    from src.security.security import RBACManager, AuditLogger, User, Role
    from src.security.pm_gate import PMGateController, GateDimension

    # 创建带 RBAC 的 gate
    audit = AuditLogger(log_dir=tempfile.mkdtemp(prefix="qlib_test_"))
    rbac = RBACManager(audit_logger=audit)

    # 注册用户
    rbac.add_user(User(user_id="pm_zhang", name="PM Zhang", role=Role.PORTFOLIO_MANAGER))
    rbac.add_user(User(user_id="researcher_li", name="Researcher Li", role=Role.QUANT_RESEARCHER))
    rbac.add_user(User(user_id="auditor_wang", name="Auditor Wang", role=Role.COMPLIANCE_AUDITOR))

    gate = PMGateController(rbac=rbac, audit_logger=audit, min_action_interval=0)

    # 1. PM 可以操作
    action = gate.emergency_stop(
        user_id="pm_zhang",
        dimension=GateDimension.SIGNAL,
        reason="PM 测试熔断"
    )
    assert action.triggered_by == "pm_zhang"
    assert action.triggered_by_role == "portfolio_manager"
    print("  ✓ PM 成功执行一键熔断")

    # 恢复
    gate.emergency_reopen(user_id="pm_zhang", dimension=GateDimension.SIGNAL, reason="恢复")

    # 2. Researcher 无权操作
    try:
        gate.emergency_stop(
            user_id="researcher_li",
            dimension=GateDimension.SIGNAL,
            reason="研究员试图熔断"
        )
        assert False, "Researcher should not have permission"
    except PermissionError as e:
        assert "researcher_li" in str(e)
        print("  ✓ Researcher 熔断被正确拒绝")

    # 3. Auditor 无权操作
    try:
        gate.emergency_stop(
            user_id="auditor_wang",
            dimension=GateDimension.SIGNAL,
            reason="审计员试图熔断"
        )
        assert False, "Auditor should not have permission"
    except PermissionError as e:
        assert "auditor_wang" in str(e)
        print("  ✓ Compliance Auditor 熔断被正确拒绝")

    # 4. 不存在的用户
    try:
        gate.emergency_stop(
            user_id="unknown_user",
            dimension=GateDimension.SIGNAL,
            reason="不存在用户"
        )
        assert False, "Unknown user should not have permission"
    except PermissionError:
        print("  ✓ 未知用户熔断被正确拒绝")

    print("TEST 2 PASSED\n")


def test_pm_gate_global():
    """测试全局熔断"""
    print("=" * 60)
    print("TEST 3: 全局熔断与恢复")
    print("=" * 60)

    from src.security.pm_gate import PMGateController

    gate = PMGateController(min_action_interval=0)

    # 全局熔断
    actions = gate.global_emergency_stop(
        user_id="pm_admin",
        reason="系统性风险"
    )

    assert gate.can_push_signal() == False
    assert gate.can_train_model() == False
    assert gate.can_deploy_model() == False
    assert gate.is_any_closed() == True
    assert len(actions) == 2  # signal + train (deploy already closed)
    print(f"  ✓ 全局熔断成功: {len(actions)} 个维度已关闭")

    # 全局恢复
    actions = gate.global_emergency_reopen(
        user_id="pm_admin",
        reason="风险解除"
    )

    assert gate.can_push_signal() == True
    assert gate.can_train_model() == True
    assert gate.can_deploy_model() == True
    assert gate.is_any_closed() == False
    print("  ✓ 全局恢复成功: 所有维度已打开")

    # 历史记录
    history = gate.get_history()
    assert len(history) >= 5
    print(f"  ✓ 操作历史: {len(history)} 条记录")

    # 统计
    stats = gate.get_stats()
    assert stats["total_stops"] >= 2
    print(f"  ✓ 统计信息: stops={stats['total_stops']}, reopens={stats['total_reopens']}")

    print("TEST 3 PASSED\n")


def test_pm_gate_auto_trip():
    """测试自动熔断"""
    print("=" * 60)
    print("TEST 4: 自动熔断机制")
    print("=" * 60)

    from src.security.pm_gate import PMGateController, GateDimension

    gate = PMGateController(auto_trip_on_alert=True, min_action_interval=0)

    # 自动熔断
    action = gate.auto_trip(
        dimension=GateDimension.TRAIN,
        reason="PIT 数据完整性校验失败",
        source="pit_validator",
    )

    assert action is not None
    assert action.action == "auto_trip"
    assert gate.can_train_model() == False
    assert gate.can_push_signal() == True  # signal 不受影响
    print(f"  ✓ 自动熔断成功: train_gate CLOSED (source={action.metadata.get('source')})")

    # 重复自动熔断应返回 None
    action2 = gate.auto_trip(dimension=GateDimension.TRAIN, reason="重复")
    assert action2 is None
    print("  ✓ 重复自动熔断返回 None (已熔断)")

    # 禁用自动熔断
    gate2 = PMGateController(auto_trip_on_alert=False, min_action_interval=0)
    action3 = gate2.auto_trip(dimension=GateDimension.SIGNAL, reason="测试")
    assert action3 is None
    print("  ✓ 禁用自动熔断时 auto_trip 返回 None")

    print("TEST 4 PASSED\n")


def test_audit_chain():
    """测试审计日志哈希链"""
    print("=" * 60)
    print("TEST 5: 审计日志防篡改哈希链")
    print("=" * 60)

    from src.security.security import AuditLogger

    tmp_dir = tempfile.mkdtemp(prefix="qlib_audit_")
    audit = AuditLogger(log_dir=tmp_dir)

    # 写入多条日志
    for i in range(10):
        audit.log(
            event_type="test_event",
            user="test_user",
            role="quant_researcher",
            action=f"action_{i}",
            resource=f"resource_{i}",
            detail={"seq": i},
        )

    # 验证哈希链
    result = audit.verify_chain()
    assert result["valid"] == True, f"Chain should be valid: {result}"
    assert result["chain_intact"] == True
    print(f"  ✓ 哈希链验证通过: valid={result['valid']}, total_lines={result.get('total_lines', 'N/A')}")

    # 查询日志
    entries = audit.query(event_type="test_event", limit=5)
    assert len(entries) == 5
    print(f"  ✓ 日志查询: {len(entries)} 条 (limit=5)")

    # 按用户查询
    entries = audit.query(user="test_user", limit=100)
    assert len(entries) >= 10
    print(f"  ✓ 按用户查询: {len(entries)} 条")

    # 按时间查询
    now = datetime.now(timezone.utc).isoformat()
    entries = audit.query(start_time=now, limit=100)
    assert len(entries) == 0  # 刚写入的日志时间戳肯定在 now 之前
    print("  ✓ 时间过滤正确")

    # 导出报告
    output = tmp_dir + "/test_report.json"
    path = audit.export_report(output, event_type="test_event")
    assert Path(path).exists()
    with open(path) as f:
        report = json.load(f)
    assert report["total_events"] >= 10
    print(f"  ✓ 审计报告导出: {report['total_events']} 条事件")

    # 清理
    import shutil
    shutil.rmtree(tmp_dir, ignore_errors=True)

    print("TEST 5 PASSED\n")


def test_compliance_reporter():
    """测试 SOX 合规报告生成"""
    print("=" * 60)
    print("TEST 6: SOX 合规报告生成")
    print("=" * 60)

    from src.security.security import RBACManager, AuditLogger, User, Role
    from src.security.compliance import SOXComplianceReporter, TLSEnforcer, TLSPolicy

    tmp_dir = tempfile.mkdtemp(prefix="qlib_sox_")
    audit = AuditLogger(log_dir=tmp_dir)
    rbac = RBACManager(audit_logger=audit)

    rbac.add_user(User(user_id="admin", name="Admin", role=Role.SYSTEM_ADMIN))

    tls = TLSEnforcer(policy=TLSPolicy.STRICT)
    # 写一些审计事件
    for i in range(5):
        audit.log(
            event_type="model_deploy",
            user="admin",
            role="system_admin",
            action="deploy",
            resource=f"model_v{i}",
        )

    reporter = SOXComplianceReporter(
        audit_logger=audit,
        rbac_manager=rbac,
        tls_enforcer=tls,
    )

    report = reporter.generate_quarterly_report(quarter="2026-Q2")
    assert "report_id" in report
    assert len(report["controls"]) == 7  # 7 SOX control points
    print(f"  ✓ SOX 报告已生成: {report['report_id']}")
    print(f"    总体状态: {report['overall_status']}")
    for c in report["controls"]:
        print(f"    {c['control_id']}: {c['status']}")

    # 清理
    import shutil
    shutil.rmtree(tmp_dir, ignore_errors=True)

    print("TEST 6 PASSED\n")


def test_security_exports():
    """测试 security 模块导出完整性"""
    print("=" * 60)
    print("TEST 7: Security 模块导出完整性")
    print("=" * 60)

    import src.security as sec

    # 核心安全
    assert hasattr(sec, "AES256Encryptor"), "Missing AES256Encryptor"
    assert hasattr(sec, "AuditLogger"), "Missing AuditLogger"
    assert hasattr(sec, "RBACManager"), "Missing RBACManager"
    assert hasattr(sec, "Role"), "Missing Role"
    assert hasattr(sec, "User"), "Missing User"
    assert hasattr(sec, "FineGrainedPerm"), "Missing FineGrainedPerm"

    # 合规
    assert hasattr(sec, "SOXComplianceReporter"), "Missing SOXComplianceReporter"
    assert hasattr(sec, "DiskEncryptor"), "Missing DiskEncryptor"
    assert hasattr(sec, "TLSEnforcer"), "Missing TLSEnforcer"
    assert hasattr(sec, "SecretManager"), "Missing SecretManager"

    # PM 熔断门控 (NEW)
    assert hasattr(sec, "PMGateController"), "Missing PMGateController"
    assert hasattr(sec, "GateState"), "Missing GateState"
    assert hasattr(sec, "GateDimension"), "Missing GateDimension"
    assert hasattr(sec, "GateAction"), "Missing GateAction"
    assert hasattr(sec, "GateClosedError"), "Missing GateClosedError"

    print("  ✓ 所有核心模块导出完整")
    print("  ✓ PM 熔断门控模块导出完整")
    print("TEST 7 PASSED\n")


def test_api_pydantic_models():
    """测试 API 中新增的 Pydantic 模型"""
    print("=" * 60)
    print("TEST 8: API Pydantic 模型验证")
    print("=" * 60)

    from src.api.main import (
        GateStatusResponse,
        GateActionRequest,
        GateActionResponse,
        GlobalGateActionRequest,
        AuditQueryParams,
        ComplianceReportRequest,
    )

    # GateActionRequest
    req = GateActionRequest(dimension="signal", reason="市场波动")
    assert req.dimension == "signal"
    assert req.reason == "市场波动"
    print("  ✓ GateActionRequest 验证通过")

    # GlobalGateActionRequest
    req = GlobalGateActionRequest(reason="系统性风险")
    assert req.reason == "系统性风险"
    print("  ✓ GlobalGateActionRequest 验证通过")

    # ComplianceReportRequest
    req = ComplianceReportRequest(quarter="2026-Q2")
    assert req.quarter == "2026-Q2"
    req2 = ComplianceReportRequest()
    assert req2.quarter == ""
    print("  ✓ ComplianceReportRequest 验证通过")

    # GateStatusResponse 构造
    resp = GateStatusResponse(
        gates={"signal": "open", "train": "open", "deploy": "closed"},
        can_push_signal=True,
        can_train_model=True,
        can_deploy_model=False,
        is_any_closed=True,
        stats={"total_stops": 0, "total_reopens": 0},
    )
    data = resp.model_dump() if hasattr(resp, "model_dump") else resp.dict()
    assert data["gates"]["deploy"] == "closed"
    print("  ✓ GateStatusResponse 构造通过")

    print("TEST 8 PASSED\n")


if __name__ == "__main__":
    print("\n" + "█" * 60)
    print("  T6 集成验证: RBAC PM 熔断门控 + 合规审计端点")
    print("█" * 60 + "\n")

    try:
        test_pm_gate_basic()
        test_pm_gate_rbac()
        test_pm_gate_global()
        test_pm_gate_auto_trip()
        test_audit_chain()
        test_compliance_reporter()
        test_security_exports()
        test_api_pydantic_models()

        print("=" * 60)
        print("✅ ALL T6 TESTS PASSED")
        print("=" * 60)
    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        raise
    except Exception as e:
        print(f"\n❌ UNEXPECTED ERROR: {e}")
        import traceback
        traceback.print_exc()
        raise
