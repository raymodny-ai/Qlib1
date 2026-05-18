"""
准确度校验与策略红线验证 (Accuracy Validator & Threshold Enforcement)

对标 PRD 4.2 中的严苛分析准确度与策略回测红线：
- Rank IC: 0.045 ~ 0.055
- Rank ICIR > 0.40
- Max Drawdown < -15%
- 滚动窗口交叉验证

核心组件:
- AccuracyThresholdValidator: 阈值红线校验器
- RollingICValidator: 滚动窗口 IC 稳定性验证
- DrawdownValidator: 最大回撤约束校验
- StrategyStabilityChecker: 综合策略稳定性检查
- ValidationReport: 校验报告数据结构

使用示例:
    from src.analyzers.accuracy_validator import AccuracyThresholdValidator
    
    validator = AccuracyThresholdValidator()
    report = validator.validate(
        rank_ic_series=ic_values,
        rank_icir=0.45,
        max_drawdown=-0.12,
    )
    if not report.passed:
        raise AccuracyViolationError(report)
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from src.utils.logger import get_logger


# ============================================================================
#  阈值常量 (PRD 4.2 红线)
# ============================================================================

class ThresholdPreset:
    """PRD 4.2 预定义阈值"""
    # IC 阈值
    RANK_IC_MIN: float = 0.045
    RANK_IC_MAX: float = 0.055
    RANK_ICIR_MIN: float = 0.40

    # 回撤阈值
    MAX_DRAWDOWN_LIMIT: float = -0.15  # 绝对值 < 15%

    # 滚动窗口
    ROLLING_WINDOW_MONTHS: int = 12

    # IC 稳定性
    IC_STD_MAX: float = 0.08       # IC 标准差上限
    IC_POSITIVE_RATIO_MIN: float = 0.55  # IC 正值比率下限

    # 夏普比率
    SHARPE_MIN: float = 0.5
    ANNUAL_RETURN_MIN: float = 0.05


# ============================================================================
#  数据结构
# ============================================================================

class CheckSeverity(str, Enum):
    """校验严重级别"""
    PASS = "pass"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass
class ThresholdCheck:
    """单条红线检查结果"""
    check_name: str
    severity: CheckSeverity = CheckSeverity.PASS
    actual_value: float = 0.0
    expected_range: str = ""
    message: str = ""
    detail: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ValidationReport:
    """校验报告"""
    passed: bool = True
    checks: List[ThresholdCheck] = field(default_factory=list)
    summary: str = ""

    @property
    def critical_count(self) -> int:
        return sum(1 for c in self.checks if c.severity == CheckSeverity.CRITICAL)

    @property
    def warning_count(self) -> int:
        return sum(1 for c in self.checks if c.severity == CheckSeverity.WARNING)

    @property
    def passing_count(self) -> int:
        return sum(1 for c in self.checks if c.severity == CheckSeverity.PASS)

    def add_check(self, check: ThresholdCheck):
        self.checks.append(check)
        if check.severity == CheckSeverity.CRITICAL:
            self.passed = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "passed": self.passed,
            "critical_count": self.critical_count,
            "warning_count": self.warning_count,
            "summary": self.summary,
            "checks": [
                {
                    "name": c.check_name,
                    "severity": c.severity.value,
                    "actual": c.actual_value,
                    "expected": c.expected_range,
                    "message": c.message,
                }
                for c in self.checks
            ],
        }


# ============================================================================
#  基础阈值校验器
# ============================================================================

class AccuracyThresholdValidator:
    """
    PRD 4.2 准确度红线校验器

    验证 IC / ICIR / Drawdown 是否满足 PRD 红线要求。
    """

    def __init__(
        self,
        rank_ic_min: float = ThresholdPreset.RANK_IC_MIN,
        rank_ic_max: float = ThresholdPreset.RANK_IC_MAX,
        rank_icir_min: float = ThresholdPreset.RANK_ICIR_MIN,
        max_drawdown_limit: float = ThresholdPreset.MAX_DRAWDOWN_LIMIT,
        ic_std_max: float = ThresholdPreset.IC_STD_MAX,
        ic_positive_ratio_min: float = ThresholdPreset.IC_POSITIVE_RATIO_MIN,
    ):
        self.rank_ic_min = rank_ic_min
        self.rank_ic_max = rank_ic_max
        self.rank_icir_min = rank_icir_min
        self.max_drawdown_limit = max_drawdown_limit
        self.ic_std_max = ic_std_max
        self.ic_positive_ratio_min = ic_positive_ratio_min
        self.logger = get_logger()

    def validate(
        self,
        rank_ic_mean: float = 0.0,
        rank_ic_series: Optional[np.ndarray] = None,
        rank_icir: float = 0.0,
        ic_mean: float = 0.0,
        ic_series: Optional[np.ndarray] = None,
        icir: float = 0.0,
        max_drawdown: float = 0.0,
        sharpe_ratio: Optional[float] = None,
        annual_return: Optional[float] = None,
    ) -> ValidationReport:
        """
        执行全面红线校验

        Args:
            rank_ic_mean: 平均 Rank IC
            rank_ic_series: Rank IC 时序
            rank_icir: Rank ICIR
            ic_mean: 平均 Pearson IC
            ic_series: Pearson IC 时序
            icir: Pearson ICIR
            max_drawdown: 最大回撤 (负数)
            sharpe_ratio: 夏普比率
            annual_return: 年化收益率

        Returns:
            ValidationReport
        """
        report = ValidationReport()

        # --- Rank IC 范围校验 ---
        if rank_ic_mean:
            self._check_rank_ic_range(report, rank_ic_mean)
        elif rank_ic_series is not None and len(rank_ic_series) > 0:
            self._check_rank_ic_range(report, float(np.mean(rank_ic_series)))

        # --- Rank ICIR 校验 ---
        if rank_icir != 0.0:
            self._check_rank_icir(report, rank_icir)

        # --- IC 标准差 ---
        if ic_series is not None and len(ic_series) > 1:
            self._check_ic_stability(report, ic_series)

        if rank_ic_series is not None and len(rank_ic_series) > 1:
            self._check_rank_ic_stability(report, rank_ic_series)

        # --- 最大回撤校验 ---
        if max_drawdown != 0.0:
            self._check_drawdown(report, max_drawdown)

        # --- 可选检查 ---
        if sharpe_ratio is not None:
            self._check_sharpe(report, sharpe_ratio)

        if annual_return is not None:
            self._check_annual_return(report, annual_return)

        # 生成摘要
        report.summary = self._build_summary(report)
        self.logger.info(
            "红线校验完成",
            passed=report.passed,
            critical=report.critical_count,
            warning=report.warning_count,
        )
        return report

    def validate_simple(
        self,
        rank_ic: float,
        rank_icir: float,
        max_drawdown: float,
    ) -> Tuple[bool, str]:
        """
        快速三要素校验

        Returns:
            (passed, message)
        """
        issues = []

        if rank_ic < self.rank_ic_min:
            issues.append(f"Rank IC {rank_ic:.4f} < {self.rank_ic_min}")
        elif rank_ic > self.rank_ic_max:
            issues.append(f"Rank IC {rank_ic:.4f} > {self.rank_ic_max}")

        if rank_icir < self.rank_icir_min:
            issues.append(f"Rank ICIR {rank_icir:.3f} < {self.rank_icir_min}")

        if max_drawdown < self.max_drawdown_limit:
            issues.append(f"MaxDD {max_drawdown:.2%} < {self.max_drawdown_limit:.2%}")

        if issues:
            return False, "; ".join(issues)
        return True, "All thresholds passed"

    # ------------------------------------------------------------------
    #  单项检查
    # ------------------------------------------------------------------

    def _check_rank_ic_range(self, report: ValidationReport, value: float):
        if value < self.rank_ic_min:
            report.add_check(ThresholdCheck(
                check_name="Rank IC 下限",
                severity=CheckSeverity.CRITICAL,
                actual_value=round(value, 6),
                expected_range=f"[{self.rank_ic_min}, {self.rank_ic_max}]",
                message=f"Rank IC {value:.4f} 低于红线 {self.rank_ic_min}",
            ))
        elif value > self.rank_ic_max:
            report.add_check(ThresholdCheck(
                check_name="Rank IC 上限",
                severity=CheckSeverity.WARNING,
                actual_value=round(value, 6),
                expected_range=f"[{self.rank_ic_min}, {self.rank_ic_max}]",
                message=f"Rank IC {value:.4f} 高于上限 {self.rank_ic_max} (可能过拟合)",
            ))
        else:
            report.add_check(ThresholdCheck(
                check_name="Rank IC 范围",
                severity=CheckSeverity.PASS,
                actual_value=round(value, 6),
                expected_range=f"[{self.rank_ic_min}, {self.rank_ic_max}]",
                message="Rank IC 在目标区间内",
            ))

    def _check_rank_icir(self, report: ValidationReport, value: float):
        if value < self.rank_icir_min:
            report.add_check(ThresholdCheck(
                check_name="Rank ICIR",
                severity=CheckSeverity.CRITICAL,
                actual_value=round(value, 4),
                expected_range=f"> {self.rank_icir_min}",
                message=f"Rank ICIR {value:.3f} 低于红线 {self.rank_icir_min}",
            ))
        else:
            report.add_check(ThresholdCheck(
                check_name="Rank ICIR",
                severity=CheckSeverity.PASS,
                actual_value=round(value, 4),
                expected_range=f"> {self.rank_icir_min}",
                message="Rank ICIR 达标",
            ))

    def _check_drawdown(self, report: ValidationReport, value: float):
        # value 应为负数，如 -0.12
        if value < self.max_drawdown_limit:  # 更负，超出回撤限制
            report.add_check(ThresholdCheck(
                check_name="Max Drawdown",
                severity=CheckSeverity.CRITICAL,
                actual_value=round(value, 6),
                expected_range=f"> {self.max_drawdown_limit}",
                message=f"最大回撤 {value:.2%} 突破红线 {self.max_drawdown_limit:.2%}",
            ))
        else:
            report.add_check(ThresholdCheck(
                check_name="Max Drawdown",
                severity=CheckSeverity.PASS,
                actual_value=round(value, 6),
                expected_range=f"> {self.max_drawdown_limit}",
                message="回撤在安全范围内",
            ))

    def _check_ic_stability(self, report: ValidationReport, ic_series: np.ndarray):
        """IC 标准差检验"""
        ic_std = float(np.std(ic_series, ddof=1))
        if ic_std > self.ic_std_max:
            report.add_check(ThresholdCheck(
                check_name="IC 标准差",
                severity=CheckSeverity.WARNING,
                actual_value=round(ic_std, 6),
                expected_range=f"≤ {self.ic_std_max}",
                message=f"IC 标准差 {ic_std:.4f} > {self.ic_std_max} (信号不稳定)",
            ))
        else:
            report.add_check(ThresholdCheck(
                check_name="IC 标准差",
                severity=CheckSeverity.PASS,
                actual_value=round(ic_std, 6),
                expected_range=f"≤ {self.ic_std_max}",
                message="IC 标准差正常",
            ))

    def _check_rank_ic_stability(self, report: ValidationReport, rank_ic_series: np.ndarray):
        """Rank IC 正值比率检验"""
        positive_ratio = float(np.mean(rank_ic_series > 0))
        if positive_ratio < self.ic_positive_ratio_min:
            report.add_check(ThresholdCheck(
                check_name="IC 正值比率",
                severity=CheckSeverity.WARNING,
                actual_value=round(positive_ratio, 4),
                expected_range=f"> {self.ic_positive_ratio_min}",
                message=f"IC 正值比率 {positive_ratio:.2%} < {self.ic_positive_ratio_min:.0%}",
            ))
        else:
            report.add_check(ThresholdCheck(
                check_name="IC 正值比率",
                severity=CheckSeverity.PASS,
                actual_value=round(positive_ratio, 4),
                expected_range=f"> {self.ic_positive_ratio_min}",
                message="IC 正值比率正常",
            ))

    def _check_sharpe(self, report: ValidationReport, value: float):
        if value < ThresholdPreset.SHARPE_MIN:
            report.add_check(ThresholdCheck(
                check_name="夏普比率",
                severity=CheckSeverity.WARNING,
                actual_value=round(value, 4),
                expected_range=f"> {ThresholdPreset.SHARPE_MIN}",
                message=f"夏普比率 {value:.3f} < {ThresholdPreset.SHARPE_MIN}",
            ))
        else:
            report.add_check(ThresholdCheck(
                check_name="夏普比率",
                severity=CheckSeverity.PASS,
                actual_value=round(value, 4),
                expected_range=f"> {ThresholdPreset.SHARPE_MIN}",
            ))

    def _check_annual_return(self, report: ValidationReport, value: float):
        if value < ThresholdPreset.ANNUAL_RETURN_MIN:
            report.add_check(ThresholdCheck(
                check_name="年化收益率",
                severity=CheckSeverity.WARNING,
                actual_value=round(value, 4),
                expected_range=f"> {ThresholdPreset.ANNUAL_RETURN_MIN}",
                message=f"年化收益率 {value:.2%} < {ThresholdPreset.ANNUAL_RETURN_MIN:.0%}",
            ))
        else:
            report.add_check(ThresholdCheck(
                check_name="年化收益率",
                severity=CheckSeverity.PASS,
                actual_value=round(value, 4),
                expected_range=f"> {ThresholdPreset.ANNUAL_RETURN_MIN}",
            ))

    def _build_summary(self, report: ValidationReport) -> str:
        if report.passed:
            return f"All {len(report.checks)} checks passed"
        parts = []
        if report.critical_count:
            parts.append(f"{report.critical_count} CRITICAL")
        if report.warning_count:
            parts.append(f"{report.warning_count} WARNING")
        return f"FAILED: {', '.join(parts)}"


# ============================================================================
#  滚动窗口 IC 校验器
# ============================================================================

@dataclass
class RollingWindowResult:
    """滚动窗口结果"""
    window_size: int
    ic_series: np.ndarray
    ic_mean: float
    ic_std: float
    ic_min: float
    ic_max: float
    ic_stable_ratio: float  # 在目标区间内的窗口比例
    violation_periods: List[Tuple[int, float]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "window_size": self.window_size,
            "ic_mean": round(self.ic_mean, 6),
            "ic_std": round(self.ic_std, 6),
            "ic_min": round(self.ic_min, 6),
            "ic_max": round(self.ic_max, 6),
            "ic_stable_ratio": round(self.ic_stable_ratio, 4),
            "violation_count": len(self.violation_periods),
        }


class RollingICValidator:
    """
    滚动窗口 IC 稳定性验证器

    对标 PRD 4.2: 在基于标准美股测试集的滚动窗口验证中，
    平均 Rank IC 必须稳定保持在 0.045 ~ 0.055 区间。
    """

    def __init__(
        self,
        window_size: int = 60,  # 约3个月交易日
        ic_min: float = ThresholdPreset.RANK_IC_MIN,
        ic_max: float = ThresholdPreset.RANK_IC_MAX,
    ):
        self.window_size = window_size
        self.ic_min = ic_min
        self.ic_max = ic_max
        self.logger = get_logger()

    def validate(
        self,
        ic_series: np.ndarray,
        dates: Optional[List[str]] = None,
    ) -> RollingWindowResult:
        """
        执行滚动窗口 IC 验证

        Args:
            ic_series: IC 时序 (每期一个值)
            dates: 对应日期列表 (可选)

        Returns:
            RollingWindowResult
        """
        ic_series = np.asarray(ic_series, dtype=np.float64)
        ic_series = ic_series[~np.isnan(ic_series)]

        if len(ic_series) < self.window_size:
            self.logger.warning(
                "IC 序列长度不足",
                length=len(ic_series),
                window=self.window_size,
            )
            return RollingWindowResult(
                window_size=self.window_size,
                ic_series=np.array([]),
                ic_mean=float(np.mean(ic_series)),
                ic_std=float(np.std(ic_series, ddof=1)),
                ic_min=float(np.min(ic_series)),
                ic_max=float(np.max(ic_series)),
                ic_stable_ratio=0.0,
            )

        n_windows = len(ic_series) - self.window_size + 1
        window_means = np.array([
            np.mean(ic_series[i:i + self.window_size])
            for i in range(n_windows)
        ])

        # 检测违规窗口
        violation_periods = []
        stable_count = 0
        for i, wm in enumerate(window_means):
            if self.ic_min <= wm <= self.ic_max:
                stable_count += 1
            else:
                violation_periods.append((i, wm))

        result = RollingWindowResult(
            window_size=self.window_size,
            ic_series=window_means,
            ic_mean=float(np.mean(window_means)),
            ic_std=float(np.std(window_means, ddof=1)),
            ic_min=float(np.min(window_means)),
            ic_max=float(np.max(window_means)),
            ic_stable_ratio=stable_count / n_windows if n_windows > 0 else 0.0,
            violation_periods=violation_periods,
        )

        self.logger.info(
            "滚动窗口 IC 验证完成",
            stable_ratio=round(result.ic_stable_ratio, 4),
            violations=len(violation_periods),
        )
        return result

    def is_stable(self, ic_series: np.ndarray, min_stable_ratio: float = 0.70) -> bool:
        """
        快速判断 IC 是否稳定

        Args:
            ic_series: IC 时序
            min_stable_ratio: 最小稳定窗口比例 (默认 70%)
        """
        result = self.validate(ic_series)
        return result.ic_stable_ratio >= min_stable_ratio


# ============================================================================
#  回撤约束校验器
# ============================================================================

class DrawdownValidator:
    """
    最大回撤约束校验器

    对标 PRD 4.2: 核心策略在极端市场恐慌时期的最大历史回撤
    必须被硬性约束在 -15% 以内。
    """

    def __init__(self, max_drawdown_limit: float = ThresholdPreset.MAX_DRAWDOWN_LIMIT):
        self.max_drawdown_limit = max_drawdown_limit  # 如 -0.15
        self.logger = get_logger()

    @staticmethod
    def compute_drawdowns(nav_series: np.ndarray) -> np.ndarray:
        """
        从净值曲线计算回撤序列

        Args:
            nav_series: 净值序列 (1D array)

        Returns:
            回撤序列 (负数), 如 [-0.02, 0, -0.05, ...]
        """
        nav = np.asarray(nav_series, dtype=np.float64)
        cummax = np.maximum.accumulate(nav)
        drawdowns = (nav - cummax) / cummax
        return drawdowns

    @staticmethod
    def compute_max_drawdown(nav_series: np.ndarray) -> float:
        """
        计算最大回撤

        Args:
            nav_series: 净值序列

        Returns:
            最大回撤值 (负数，如 -0.12)
        """
        drawdowns = DrawdownValidator.compute_drawdowns(nav_series)
        return float(np.min(drawdowns))

    @staticmethod
    def compute_drawdown_duration(nav_series: np.ndarray) -> int:
        """
        计算最大回撤持续天数

        Returns:
            最长连续亏损天数
        """
        drawdowns = DrawdownValidator.compute_drawdowns(nav_series)
        in_drawdown = drawdowns < 0
        if not np.any(in_drawdown):
            return 0

        # 找最长连续 True 段
        max_duration = 0
        current = 0
        for x in in_drawdown:
            if x:
                current += 1
                max_duration = max(max_duration, current)
            else:
                current = 0
        return max_duration

    def validate(self, nav_series: np.ndarray) -> ValidationReport:
        """
        校验回撤是否在红线内

        Args:
            nav_series: 净值曲线

        Returns:
            ValidationReport
        """
        report = ValidationReport()
        max_dd = self.compute_max_drawdown(nav_series)
        duration = self.compute_drawdown_duration(nav_series)

        report.add_check(ThresholdCheck(
            check_name="Max Drawdown",
            severity=CheckSeverity.CRITICAL if max_dd < self.max_drawdown_limit else CheckSeverity.PASS,
            actual_value=round(max_dd, 6),
            expected_range=f"> {self.max_drawdown_limit}",
            message=(
                f"最大回撤 {max_dd:.2%} 突破红线 {self.max_drawdown_limit:.2%}"
                if max_dd < self.max_drawdown_limit
                else f"最大回撤 {max_dd:.2%} 在安全范围内"
            ),
            detail={"max_drawdown_duration": duration},
        ))

        report.summary = "回撤校验通过" if report.passed else f"回撤超标: {max_dd:.2%}"
        return report


# ============================================================================
#  策略稳定性综合检查器
# ============================================================================

class StrategyStabilityChecker:
    """
    策略稳定性综合检查器

    整合 IC 校验、回撤校验、滚动窗口验证，提供一站式策略红线审核。

    使用示例:
        checker = StrategyStabilityChecker()
        result = checker.check(
            rank_ic_series=ic_daily,
            rank_icir=0.45,
            nav_series=nav,
            returns_series=daily_returns,
        )
        if result["approved"]:
            deploy_signals(checker)
    """

    def __init__(self):
        self.ic_validator = AccuracyThresholdValidator()
        self.rolling_validator = RollingICValidator()
        self.drawdown_validator = DrawdownValidator()
        self.logger = get_logger()

    def check(
        self,
        rank_ic_series: np.ndarray,
        rank_icir: float,
        nav_series: Optional[np.ndarray] = None,
        returns_series: Optional[np.ndarray] = None,
        sharpe_ratio: Optional[float] = None,
        min_stable_ratio: float = 0.70,
    ) -> Dict[str, Any]:
        """
        综合策略稳定性检查

        Args:
            rank_ic_series: Rank IC 日序列
            rank_icir: Rank ICIR
            nav_series: 净值曲线
            returns_series: 日收益率 (用于计算夏普)
            sharpe_ratio: 预计算的夏普比率
            min_stable_ratio: 滚动窗口稳定比例阈值

        Returns:
            {
                "approved": bool,
                "issues": [...],
                "ic_report": {...},
                "rolling_report": {...},
                "drawdown_report": {...},
            }
        """
        result: Dict[str, Any] = {
            "approved": True,
            "issues": [],
        }

        # 1. IC 阈值校验
        rank_ic_mean = float(np.mean(rank_ic_series))
        ic_report = self.ic_validator.validate(
            rank_ic_mean=rank_ic_mean,
            rank_ic_series=rank_ic_series,
            rank_icir=rank_icir,
            max_drawdown=0.0,
        )
        result["ic_report"] = ic_report.to_dict()
        if not ic_report.passed:
            result["approved"] = False
            result["issues"].extend([
                c.message for c in ic_report.checks
                if c.severity == CheckSeverity.CRITICAL
            ])

        # 2. 滚动窗口稳定性
        rolling_result = self.rolling_validator.validate(rank_ic_series)
        result["rolling_report"] = rolling_result.to_dict()
        if rolling_result.ic_stable_ratio < min_stable_ratio:
            result["approved"] = False
            result["issues"].append(
                f"滚动窗口稳定率 {rolling_result.ic_stable_ratio:.1%} < {min_stable_ratio:.0%}"
            )

        # 3. 回撤校验 (如果有净值曲线)
        if nav_series is not None and len(nav_series) > 0:
            dd_report = self.drawdown_validator.validate(nav_series)
            result["drawdown_report"] = dd_report.to_dict()
            if not dd_report.passed:
                result["approved"] = False
                result["issues"].append(f"回撤超标: {dd_report.summary}")

        # 4. 夏普比率 (从收益率计算或使用预计算值)
        if sharpe_ratio is None and returns_series is not None and len(returns_series) > 0:
            daily_returns = np.asarray(returns_series, dtype=np.float64)
            daily_returns = daily_returns[~np.isnan(daily_returns)]
            if len(daily_returns) > 0:
                mean_ret = np.mean(daily_returns)
                std_ret = np.std(daily_returns, ddof=1)
                sharpe_ratio = float(mean_ret / (std_ret + 1e-12) * np.sqrt(252))

        if sharpe_ratio is not None:
            if sharpe_ratio < ThresholdPreset.SHARPE_MIN:
                result["issues"].append(f"夏普比率 {sharpe_ratio:.3f} < {ThresholdPreset.SHARPE_MIN}")

        self.logger.info(
            "策略稳定性检查完成",
            approved=result["approved"],
            issues_count=len(result["issues"]),
        )
        return result


# ============================================================================
#  便捷函数
# ============================================================================

def quick_validate(
    rank_ic: float,
    rank_icir: float,
    max_drawdown: float,
) -> Tuple[bool, str]:
    """快速三要素校验 (无报告开销)"""
    validator = AccuracyThresholdValidator()
    return validator.validate_simple(rank_ic, rank_icir, max_drawdown)
