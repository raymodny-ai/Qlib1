"""
第4章 准确度校验与策略红线验证 单元测试

覆盖:
- AccuracyThresholdValidator: IC/ICIR/Drawdown/夏普/年化 红线校验
- RollingICValidator: 滚动窗口 IC 稳定性验证
- DrawdownValidator: 回撤计算与约束校验
- StrategyStabilityChecker: 综合策略稳定性检查
- ValidationReport / ThresholdCheck: 数据结构
- quick_validate: 便捷函数
"""

import pytest
import numpy as np

from src.analyzers.accuracy_validator import (
    AccuracyThresholdValidator,
    RollingICValidator,
    RollingWindowResult,
    DrawdownValidator,
    StrategyStabilityChecker,
    ValidationReport,
    ThresholdCheck,
    CheckSeverity,
    ThresholdPreset,
    quick_validate,
)


# ============================================================================
#  AccuracyThresholdValidator 测试
# ============================================================================

class TestAccuracyThresholdValidator:
    """AccuracyThresholdValidator 红线校验器测试"""

    def test_rank_ic_in_range(self):
        """Rank IC 在目标区间内"""
        validator = AccuracyThresholdValidator()
        report = validator.validate(
            rank_ic_mean=0.050,
            rank_icir=0.50,
            max_drawdown=-0.10,
        )
        assert report.passed is True
        assert report.critical_count == 0

    def test_rank_ic_below_threshold(self):
        """Rank IC 低于红线"""
        validator = AccuracyThresholdValidator()
        report = validator.validate(rank_ic_mean=0.030, rank_icir=0.50, max_drawdown=-0.10)
        assert report.passed is False
        assert report.critical_count == 1
        assert any("Rank IC 下限" in c.check_name for c in report.checks if c.severity == CheckSeverity.CRITICAL)

    def test_rank_ic_above_upper(self):
        """Rank IC 高于上限 (可能过拟合)"""
        validator = AccuracyThresholdValidator()
        report = validator.validate(rank_ic_mean=0.065, rank_icir=0.50, max_drawdown=-0.10)
        # 上限只是 WARNING，不应导致 failed
        assert report.passed is True
        assert any(c.severity == CheckSeverity.WARNING for c in report.checks)

    def test_rank_icir_below_threshold(self):
        """Rank ICIR 低于红线"""
        validator = AccuracyThresholdValidator()
        report = validator.validate(rank_ic_mean=0.050, rank_icir=0.25, max_drawdown=-0.10)
        assert report.passed is False
        assert any("Rank ICIR" in c.check_name for c in report.checks if c.severity == CheckSeverity.CRITICAL)

    def test_rank_icir_passing(self):
        """Rank ICIR 达标"""
        validator = AccuracyThresholdValidator()
        report = validator.validate(rank_ic_mean=0.050, rank_icir=0.50, max_drawdown=-0.10)
        assert any("Rank ICIR" in c.check_name and c.severity == CheckSeverity.PASS for c in report.checks)

    def test_drawdown_within_limit(self):
        """回撤在安全范围内"""
        validator = AccuracyThresholdValidator()
        report = validator.validate(rank_ic_mean=0.050, rank_icir=0.50, max_drawdown=-0.10)
        assert any("Max Drawdown" in c.check_name and c.severity == CheckSeverity.PASS for c in report.checks)

    def test_drawdown_breach(self):
        """回撤突破红线"""
        validator = AccuracyThresholdValidator()
        report = validator.validate(rank_ic_mean=0.050, rank_icir=0.50, max_drawdown=-0.20)
        assert report.passed is False
        assert any("Max Drawdown" in c.check_name and c.severity == CheckSeverity.CRITICAL for c in report.checks)

    def test_drawdown_zero_skipped(self):
        """drawdown=0 时跳过校验"""
        validator = AccuracyThresholdValidator()
        report = validator.validate(rank_ic_mean=0.050, rank_icir=0.50, max_drawdown=0.0)
        assert report.passed is True
        # drawdown 为 0 时不添加 check
        assert not any("Drawdown" in c.check_name for c in report.checks)

    def test_ic_stability_warning(self):
        """IC 标准差超标"""
        validator = AccuracyThresholdValidator()
        # 生成高波动 IC 序列 (Pearson IC → 触发 _check_ic_stability)
        ic_series = np.array([0.02, 0.15, -0.05, 0.20, 0.01, -0.10, 0.18, -0.08])
        report = validator.validate(ic_series=ic_series, rank_icir=0.50, max_drawdown=-0.10)
        assert any("IC 标准差" in c.check_name for c in report.checks)

    def test_ic_positive_ratio_low(self):
        """IC 正值比率过低"""
        validator = AccuracyThresholdValidator()
        # 大部分为负值的 Rank IC
        rank_ic = np.array([-0.01, -0.02, 0.01, -0.03, 0.005, -0.01, -0.02, -0.01, 0.01, -0.015])
        report = validator.validate(rank_ic_series=rank_ic, rank_icir=0.50, max_drawdown=-0.10)
        assert any("IC 正值比率" in c.check_name for c in report.checks)

    def test_rank_ic_from_series(self):
        """从序列计算 Rank IC 均值"""
        validator = AccuracyThresholdValidator()
        rank_ic_series = np.array([0.048, 0.052, 0.050, 0.049, 0.051])
        report = validator.validate(rank_ic_series=rank_ic_series, rank_icir=0.50, max_drawdown=-0.10)
        assert any("Rank IC 范围" in c.check_name and c.severity == CheckSeverity.PASS for c in report.checks)

    def test_sharpe_warning(self):
        """夏普比率不足"""
        validator = AccuracyThresholdValidator()
        report = validator.validate(rank_ic_mean=0.050, rank_icir=0.50, max_drawdown=-0.10, sharpe_ratio=0.3)
        assert any("夏普比率" in c.check_name and c.severity == CheckSeverity.WARNING for c in report.checks)

    def test_annual_return_warning(self):
        """年化收益率不足"""
        validator = AccuracyThresholdValidator()
        report = validator.validate(rank_ic_mean=0.050, rank_icir=0.50, max_drawdown=-0.10, annual_return=0.02)
        assert any("年化收益率" in c.check_name and c.severity == CheckSeverity.WARNING for c in report.checks)

    def test_all_thresholds_passing(self):
        """全部阈值通过"""
        validator = AccuracyThresholdValidator()
        report = validator.validate(
            rank_ic_mean=0.050,
            rank_icir=0.50,
            max_drawdown=-0.10,
            sharpe_ratio=0.8,
            annual_return=0.12,
        )
        assert report.passed is True
        assert len(report.checks) > 0
        assert all(c.severity == CheckSeverity.PASS for c in report.checks)

    def test_validate_simple_passing(self):
        """快速校验通过"""
        validator = AccuracyThresholdValidator()
        passed, msg = validator.validate_simple(0.050, 0.50, -0.10)
        assert passed is True
        assert "passed" in msg

    def test_validate_simple_failing_all(self):
        """快速校验三要素同时失败"""
        validator = AccuracyThresholdValidator()
        passed, msg = validator.validate_simple(0.030, 0.20, -0.25)
        assert passed is False
        assert "Rank IC" in msg
        assert "Rank ICIR" in msg
        assert "MaxDD" in msg

    def test_custom_thresholds(self):
        """自定义阈值"""
        validator = AccuracyThresholdValidator(
            rank_ic_min=0.03,
            rank_ic_max=0.08,
            rank_icir_min=0.30,
            max_drawdown_limit=-0.20,
        )
        report = validator.validate(rank_ic_mean=0.035, rank_icir=0.35, max_drawdown=-0.12)
        assert report.passed is True

    def test_report_to_dict(self):
        """ValidationReport to_dict"""
        report = ValidationReport(passed=True)
        report.add_check(ThresholdCheck(
            check_name="TestCheck",
            severity=CheckSeverity.PASS,
            actual_value=0.05,
            expected_range="[0, 1]",
        ))
        d = report.to_dict()
        assert d["passed"] is True
        assert len(d["checks"]) == 1
        assert d["checks"][0]["name"] == "TestCheck"


# ============================================================================
#  RollingICValidator 测试
# ============================================================================

class TestRollingICValidator:
    """RollingICValidator 滚动窗口 IC 验证器测试"""

    def test_validate_stable_ic(self):
        """稳定 IC 序列"""
        validator = RollingICValidator(window_size=20)
        # 稳定的 0.05 附近 IC
        np.random.seed(42)
        ic_series = np.random.normal(0.05, 0.005, 200)
        result = validator.validate(ic_series)
        assert isinstance(result, RollingWindowResult)
        assert result.window_size == 20
        assert result.ic_stable_ratio > 0.5

    def test_validate_unstable_ic(self):
        """不稳定 IC 序列"""
        validator = RollingICValidator(window_size=20)
        np.random.seed(42)
        # 高波动、均值不在目标区间的 IC
        ic_series = np.random.normal(0.02, 0.03, 200)
        result = validator.validate(ic_series)
        assert result.ic_stable_ratio < 0.5

    def test_is_stable_true(self):
        """is_stable 快速判断 - 稳定"""
        validator = RollingICValidator(window_size=20)
        np.random.seed(42)
        ic_series = np.random.normal(0.05, 0.003, 300)
        assert validator.is_stable(ic_series, min_stable_ratio=0.5)

    def test_is_stable_false(self):
        """is_stable 快速判断 - 不稳定"""
        validator = RollingICValidator(window_size=20)
        np.random.seed(42)
        ic_series = np.random.normal(0.02, 0.04, 300)
        assert not validator.is_stable(ic_series, min_stable_ratio=0.5)

    def test_short_series(self):
        """序列长度不足"""
        validator = RollingICValidator(window_size=60)
        ic_series = np.array([0.05] * 30)
        result = validator.validate(ic_series)
        assert len(result.ic_series) == 0
        assert result.ic_stable_ratio == 0.0

    def test_nan_handling(self):
        """NaN 处理"""
        validator = RollingICValidator(window_size=10)
        ic_series = np.array([0.05] * 30 + [np.nan] * 5 + [0.05] * 30)
        result = validator.validate(ic_series)
        assert result.window_size == 10

    def test_to_dict(self):
        """RollingWindowResult to_dict"""
        result = RollingWindowResult(
            window_size=20,
            ic_series=np.array([0.048, 0.052, 0.050]),
            ic_mean=0.05,
            ic_std=0.002,
            ic_min=0.048,
            ic_max=0.052,
            ic_stable_ratio=0.95,
            violation_periods=[(5, 0.03)],
        )
        d = result.to_dict()
        assert d["window_size"] == 20
        assert d["violation_count"] == 1

    def test_custom_bounds(self):
        """自定义 IC 区间"""
        validator = RollingICValidator(window_size=10, ic_min=0.03, ic_max=0.07)
        np.random.seed(42)
        ic_series = np.random.normal(0.05, 0.01, 100)
        result = validator.validate(ic_series)
        assert result.ic_stable_ratio > 0.5


# ============================================================================
#  DrawdownValidator 测试
# ============================================================================

class TestDrawdownValidator:
    """DrawdownValidator 回撤校验器测试"""

    def test_compute_max_drawdown_simple(self):
        """简单回撤计算"""
        nav = np.array([1.0, 1.1, 0.9, 1.0, 1.2])
        mdd = DrawdownValidator.compute_max_drawdown(nav)
        # 最大回撤出现在 1.1 → 0.9: (0.9-1.1)/1.1 = -0.1818...
        assert mdd == pytest.approx(-0.181818, rel=0.01)

    def test_compute_max_drawdown_no_drawdown(self):
        """无回撤"""
        nav = np.array([1.0, 1.1, 1.2, 1.3])
        mdd = DrawdownValidator.compute_max_drawdown(nav)
        assert mdd == 0.0

    def test_compute_drawdown_duration(self):
        """回撤持续时间"""
        nav = np.array([1.0, 0.9, 0.8, 0.85, 1.0, 1.1])
        duration = DrawdownValidator.compute_drawdown_duration(nav)
        assert duration == 3  # 0.9,0.8,0.85 — below peak 1.0; 1.0 restores peak

    def test_compute_drawdown_duration_no_drawdown(self):
        """无回撤持续时间"""
        nav = np.array([1.0, 1.1, 1.2])
        duration = DrawdownValidator.compute_drawdown_duration(nav)
        assert duration == 0

    def test_compute_drawdown_series(self):
        """回撤序列"""
        nav = np.array([1.0, 1.1, 0.9, 1.0])
        dd = DrawdownValidator.compute_drawdowns(nav)
        assert dd[0] == 0.0
        assert dd[1] == 0.0
        assert dd[2] == pytest.approx(-0.181818, rel=0.01)
        assert dd[3] == pytest.approx(-0.090909, rel=0.01)

    def test_validate_passing(self):
        """回撤在安全范围内"""
        validator = DrawdownValidator(max_drawdown_limit=-0.15)
        nav = np.array([1.0, 1.05, 0.92, 1.1, 1.2])  # max DD: (0.92-1.05)/1.05 = -12.4%
        report = validator.validate(nav)
        assert report.passed is True

    def test_validate_breach(self):
        """回撤突破红线"""
        validator = DrawdownValidator(max_drawdown_limit=-0.15)
        nav = np.array([1.0, 0.8, 0.9, 1.0])  # max DD: (0.8-1)/1 = -20%
        report = validator.validate(nav)
        assert report.passed is False
        assert report.critical_count == 1

    def test_validate_includes_duration(self):
        """报告包含回撤持续时间"""
        validator = DrawdownValidator()
        nav = np.array([1.0, 0.9, 0.85, 0.8, 1.0])
        report = validator.validate(nav)
        check = report.checks[0]
        assert "max_drawdown_duration" in check.detail


# ============================================================================
#  StrategyStabilityChecker 测试
# ============================================================================

class TestStrategyStabilityChecker:
    """StrategyStabilityChecker 综合策略稳定性测试"""

    def test_check_all_passing(self):
        """全部通过"""
        checker = StrategyStabilityChecker()
        np.random.seed(42)
        rank_ic = np.random.normal(0.05, 0.005, 300)
        nav = np.array([1.0 * (1 + r) for r in np.cumsum(np.random.normal(0.0005, 0.01, 300))])

        result = checker.check(
            rank_ic_series=rank_ic,
            rank_icir=0.50,
            nav_series=nav,
        )
        assert "approved" in result
        assert "ic_report" in result
        assert "rolling_report" in result
        assert "drawdown_report" in result

    def test_check_with_sharpe(self):
        """包含夏普比率"""
        checker = StrategyStabilityChecker()
        np.random.seed(42)
        rank_ic = np.random.normal(0.05, 0.005, 300)
        returns = np.random.normal(0.0005, 0.01, 300)

        result = checker.check(
            rank_ic_series=rank_ic,
            rank_icir=0.50,
            returns_series=returns,
        )
        assert "approved" in result

    def test_check_precomputed_sharpe(self):
        """预计算夏普比率"""
        checker = StrategyStabilityChecker()
        np.random.seed(42)
        rank_ic = np.random.normal(0.05, 0.005, 300)

        result = checker.check(
            rank_ic_series=rank_ic,
            rank_icir=0.50,
            sharpe_ratio=0.8,
        )
        assert "approved" in result

    def test_check_with_empty_nav(self):
        """空净值曲线"""
        checker = StrategyStabilityChecker()
        np.random.seed(42)
        rank_ic = np.random.normal(0.05, 0.005, 300)

        result = checker.check(
            rank_ic_series=rank_ic,
            rank_icir=0.50,
            nav_series=np.array([]),
        )
        assert "drawdown_report" not in result

    def test_check_fails_on_poor_ic(self):
        """IC 不达标导致未批准"""
        checker = StrategyStabilityChecker()
        rank_ic = np.full(300, 0.02)  # 远低于 0.045

        result = checker.check(
            rank_ic_series=rank_ic,
            rank_icir=0.20,
        )
        assert result["approved"] is False
        assert len(result["issues"]) > 0

    def test_check_fails_on_drawdown_breach(self):
        """回撤超标导致未批准"""
        checker = StrategyStabilityChecker()
        np.random.seed(42)
        rank_ic = np.random.normal(0.05, 0.005, 300)
        # 制造大回撤
        nav = np.array([1.0, 0.95, 0.9, 0.8, 0.7, 0.75, 0.8])

        result = checker.check(
            rank_ic_series=rank_ic,
            rank_icir=0.50,
            nav_series=nav,
        )
        assert result["approved"] is False


# ============================================================================
#  quick_validate 测试
# ============================================================================

class TestQuickValidate:
    """quick_validate 便捷函数测试"""

    def test_passing(self):
        passed, msg = quick_validate(0.050, 0.50, -0.10)
        assert passed is True

    def test_failing(self):
        passed, msg = quick_validate(0.030, 0.20, -0.25)
        assert passed is False

    def test_failing_ic_only(self):
        passed, msg = quick_validate(0.030, 0.50, -0.10)
        assert passed is False
        assert "Rank IC" in msg


# ============================================================================
#  ThresholdPreset 测试
# ============================================================================

class TestThresholdPreset:
    """阈值预设常量测试"""

    def test_constants(self):
        assert ThresholdPreset.RANK_IC_MIN == 0.045
        assert ThresholdPreset.RANK_IC_MAX == 0.055
        assert ThresholdPreset.RANK_ICIR_MIN == 0.40
        assert ThresholdPreset.MAX_DRAWDOWN_LIMIT == -0.15
        assert ThresholdPreset.SHARPE_MIN == 0.5
