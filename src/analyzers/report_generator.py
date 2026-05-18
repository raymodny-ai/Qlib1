"""
多维绩效报告生成器 (Performance Report Generator)

Qlib 风格的分析器 (Analyzer) 与实验记录器, 将回测结果转化为
人类可读的图表报告和结构化指标。

核心组件:
- BacktestAnalyzer: 计算 IC/ICIR、累积收益、风险指标
- PerformanceReport: 生成 IC 曲线/累积收益/风险分析图表
- ReportExporter: 导出 HTML/JSON/CSV 格式报告

使用示例:
    from src.analyzers.report_generator import BacktestAnalyzer, PerformanceReport
    
    analyzer = BacktestAnalyzer()
    metrics = analyzer.compute(predictions, returns)
    report = PerformanceReport(metrics)
    report.to_html("report.html")
"""

import base64
import io
import json
import os
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

from src.utils.logger import get_logger


# ========================================================================
#  数据结构
# ========================================================================

@dataclass
class ICMetrics:
    """信息系数 (IC) 相关指标"""
    ic_series: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    rank_ic_series: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    ic_mean: float = 0.0
    ic_std: float = 0.0
    icir: float = 0.0
    rank_ic_mean: float = 0.0
    rank_ic_std: float = 0.0
    rank_icir: float = 0.0
    ic_positive_ratio: float = 0.0
    ic_significant_ratio: float = 0.0  # |IC| > 2*std 的比例

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ic_mean": round(self.ic_mean, 6),
            "ic_std": round(self.ic_std, 6),
            "icir": round(self.icir, 4),
            "rank_ic_mean": round(self.rank_ic_mean, 6),
            "rank_ic_std": round(self.rank_ic_std, 6),
            "rank_icir": round(self.rank_icir, 4),
            "ic_positive_ratio": round(self.ic_positive_ratio, 4),
            "ic_significant_ratio": round(self.ic_significant_ratio, 4),
        }


@dataclass
class ReturnMetrics:
    """收益相关指标"""
    total_return: float = 0.0
    annualized_return: float = 0.0
    annualized_volatility: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    max_drawdown_duration: int = 0
    calmar_ratio: float = 0.0
    sortino_ratio: float = 0.0
    win_rate: float = 0.0
    profit_loss_ratio: float = 0.0
    cumulative_returns: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    daily_returns: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    excess_returns: Optional[pd.Series] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_return": round(self.total_return, 6),
            "annualized_return": round(self.annualized_return, 6),
            "annualized_volatility": round(self.annualized_volatility, 6),
            "sharpe_ratio": round(self.sharpe_ratio, 4),
            "max_drawdown": round(self.max_drawdown, 6),
            "max_drawdown_duration": self.max_drawdown_duration,
            "calmar_ratio": round(self.calmar_ratio, 4),
            "sortino_ratio": round(self.sortino_ratio, 4),
            "win_rate": round(self.win_rate, 4),
            "profit_loss_ratio": round(self.profit_loss_ratio, 4),
        }


@dataclass
class RiskMetrics:
    """风险相关指标"""
    var_95: float = 0.0       # 95% VaR
    var_99: float = 0.0       # 99% VaR
    cvar_95: float = 0.0      # 95% CVaR (Expected Shortfall)
    max_daily_loss: float = 0.0
    max_daily_gain: float = 0.0
    skewness: float = 0.0
    kurtosis: float = 0.0
    stability: float = 0.0     # 收益稳定性 (正收益占比)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "var_95": round(self.var_95, 6),
            "var_99": round(self.var_99, 6),
            "cvar_95": round(self.cvar_95, 6),
            "max_daily_loss": round(self.max_daily_loss, 6),
            "max_daily_gain": round(self.max_daily_gain, 6),
            "skewness": round(self.skewness, 4),
            "kurtosis": round(self.kurtosis, 4),
            "stability": round(self.stability, 4),
        }


@dataclass
class FullReport:
    """完整绩效报告"""
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    model_name: str = ""
    ic_metrics: Optional[ICMetrics] = None
    return_metrics: Optional[ReturnMetrics] = None
    risk_metrics: Optional[RiskMetrics] = None
    extra_metrics: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        result = {
            "timestamp": self.timestamp,
            "model_name": self.model_name,
        }
        if self.ic_metrics:
            result["ic_metrics"] = self.ic_metrics.to_dict()
        if self.return_metrics:
            result["return_metrics"] = self.return_metrics.to_dict()
        if self.risk_metrics:
            result["risk_metrics"] = self.risk_metrics.to_dict()
        result.update(self.extra_metrics)
        return result


# ========================================================================
#  回测分析器
# ========================================================================

class BacktestAnalyzer:
    """
    回测分析器 — 从预测和收益数据计算全套绩效指标

    支持横截面 (cross-sectional) IC 评估和时间序列收益分析。
    """

    def __init__(self):
        self.logger = get_logger()

    def compute_ic(
        self,
        predictions: pd.DataFrame,
        returns: pd.DataFrame,
        method: str = "spearman",
    ) -> ICMetrics:
        """
        计算信息系数 (IC) 和 Rank IC

        Args:
            predictions: 预测得分 DataFrame (index=date, columns=instrument)
            returns: 真实收益率 DataFrame (同结构)
            method: 'pearson' | 'spearman'

        Returns:
            ICMetrics
        """
        common_dates = predictions.index.intersection(returns.index)

        ic_values = []
        rank_ic_values = []

        for date in common_dates:
            pred = predictions.loc[date].dropna()
            ret = returns.loc[date].dropna()

            common_inst = pred.index.intersection(ret.index)
            if len(common_inst) < 5:
                continue

            p = pred[common_inst].values
            r = ret[common_inst].values

            if method == "spearman":
                ic = scipy_stats.spearmanr(p, r)[0]
            else:
                ic = np.corrcoef(p, r)[0, 1]

            rank_ic = scipy_stats.spearmanr(p, r)[0]

            if not np.isnan(ic):
                ic_values.append(ic)
            if not np.isnan(rank_ic):
                rank_ic_values.append(rank_ic)

        ic_arr = np.array(ic_values)
        rank_ic_arr = np.array(rank_ic_values)

        if len(ic_arr) == 0:
            return ICMetrics()

        ic_mean = float(np.mean(ic_arr))
        ic_std = float(np.std(ic_arr, ddof=1))
        icir = ic_mean / (ic_std + 1e-12)
        ic_positive = float(np.mean(ic_arr > 0))
        ic_significant = float(np.mean(np.abs(ic_arr) > 2 * ic_std))

        rank_ic_mean = float(np.mean(rank_ic_arr))
        rank_ic_std = float(np.std(rank_ic_arr, ddof=1))
        rank_icir = rank_ic_mean / (rank_ic_std + 1e-12)

        return ICMetrics(
            ic_series=pd.Series(ic_arr, index=common_dates[:len(ic_arr)]),
            rank_ic_series=pd.Series(rank_ic_arr, index=common_dates[:len(rank_ic_arr)]),
            ic_mean=ic_mean,
            ic_std=ic_std,
            icir=icir,
            rank_ic_mean=rank_ic_mean,
            rank_ic_std=rank_ic_std,
            rank_icir=rank_icir,
            ic_positive_ratio=ic_positive,
            ic_significant_ratio=ic_significant,
        )

    def compute_returns(
        self,
        nav_curve: pd.Series,
        benchmark_nav: Optional[pd.Series] = None,
        risk_free_rate: float = 0.0,
    ) -> ReturnMetrics:
        """
        从 NAV 曲线计算收益指标

        Args:
            nav_curve: 净值曲线 (index=date)
            benchmark_nav: 基准净值曲线
            risk_free_rate: 无风险利率 (年化)

        Returns:
            ReturnMetrics
        """
        if len(nav_curve) < 2:
            return ReturnMetrics()

        daily_returns = nav_curve.pct_change().dropna()
        if len(daily_returns) == 0:
            return ReturnMetrics()

        # 基本收益
        total_return = nav_curve.iloc[-1] / nav_curve.iloc[0] - 1
        n_years = len(daily_returns) / 252
        annualized_return = (1 + total_return) ** (1 / max(n_years, 0.01)) - 1
        annualized_vol = float(daily_returns.std() * np.sqrt(252))

        # Sharpe
        excess_daily = daily_returns - risk_free_rate / 252
        sharpe = float(excess_daily.mean() / daily_returns.std() * np.sqrt(252)) if daily_returns.std() > 0 else 0

        # Sortino (下行波动率)
        downside_returns = daily_returns[daily_returns < 0]
        downside_std = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 1e-12
        sortino = float((daily_returns.mean() * 252 - risk_free_rate) / downside_std) if downside_std > 0 else 0

        # 最大回撤
        cummax = nav_curve.cummax()
        drawdowns = (nav_curve - cummax) / cummax
        max_dd = float(drawdowns.min())

        # 最大回撤持续期
        max_dd_duration = 0
        current_duration = 0
        dd_start = None
        for i, dd in enumerate(drawdowns):
            if dd < 0:
                if dd_start is None:
                    dd_start = i
                current_duration = i - dd_start + 1
                max_dd_duration = max(max_dd_duration, current_duration)
            else:
                dd_start = None
                current_duration = 0

        # Calmar
        calmar = annualized_return / abs(max_dd) if abs(max_dd) > 0 else 0

        # 胜率 & 盈亏比
        wins = (daily_returns > 0).sum()
        total = len(daily_returns)
        win_rate = wins / total if total > 0 else 0
        avg_win = daily_returns[daily_returns > 0].mean() if wins > 0 else 0
        avg_loss = abs(daily_returns[daily_returns < 0].mean()) if wins < total else 1e-12
        profit_loss_ratio = avg_win / avg_loss if avg_loss > 0 else 0

        # 超额收益
        excess_returns = None
        if benchmark_nav is not None:
            benchmark_returns = benchmark_nav.pct_change().dropna()
            common_idx = daily_returns.index.intersection(benchmark_returns.index)
            excess_returns = daily_returns[common_idx] - benchmark_returns[common_idx]

        return ReturnMetrics(
            total_return=total_return,
            annualized_return=annualized_return,
            annualized_volatility=annualized_vol,
            sharpe_ratio=sharpe,
            max_drawdown=max_dd,
            max_drawdown_duration=max_dd_duration,
            calmar_ratio=calmar,
            sortino_ratio=sortino,
            win_rate=win_rate,
            profit_loss_ratio=profit_loss_ratio,
            cumulative_returns=(1 + daily_returns).cumprod(),
            daily_returns=daily_returns,
            excess_returns=excess_returns,
        )

    def compute_risk(
        self,
        daily_returns: pd.Series,
    ) -> RiskMetrics:
        """
        从日收益率计算风险指标

        Args:
            daily_returns: 日收益率 Series

        Returns:
            RiskMetrics
        """
        if len(daily_returns) < 2:
            return RiskMetrics()

        rets = daily_returns.dropna().values

        # VaR
        var_95 = float(np.percentile(rets, 5))
        var_99 = float(np.percentile(rets, 1))

        # CVaR (Expected Shortfall)
        cvar_95 = float(rets[rets <= var_95].mean()) if len(rets[rets <= var_95]) > 0 else var_95

        # 极值
        max_daily_loss = float(rets.min())
        max_daily_gain = float(rets.max())

        # 偏度 & 峰度
        skewness = float(scipy_stats.skew(rets))
        kurtosis = float(scipy_stats.kurtosis(rets))

        # 稳定性
        stability = float(np.mean(rets > 0))

        return RiskMetrics(
            var_95=var_95,
            var_99=var_99,
            cvar_95=cvar_95,
            max_daily_loss=max_daily_loss,
            max_daily_gain=max_daily_gain,
            skewness=skewness,
            kurtosis=kurtosis,
            stability=stability,
        )

    def analyze(
        self,
        predictions: pd.DataFrame,
        returns: pd.DataFrame,
        nav_curve: Optional[pd.Series] = None,
        benchmark_nav: Optional[pd.Series] = None,
        model_name: str = "",
    ) -> FullReport:
        """
        一键分析 — 生成完整绩效报告

        Args:
            predictions: 预测得分 DataFrame
            returns: 真实收益率 DataFrame
            nav_curve: 策略净值曲线
            benchmark_nav: 基准净值曲线
            model_name: 模型名称

        Returns:
            FullReport
        """
        ic_metrics = self.compute_ic(predictions, returns)

        return_metrics = None
        risk_metrics = None

        if nav_curve is not None and len(nav_curve) > 1:
            return_metrics = self.compute_returns(nav_curve, benchmark_nav)
            risk_metrics = self.compute_risk(return_metrics.daily_returns)

        return FullReport(
            model_name=model_name,
            ic_metrics=ic_metrics,
            return_metrics=return_metrics,
            risk_metrics=risk_metrics,
        )


# ========================================================================
#  绩效报告生成器 (图表)
# ========================================================================

class PerformanceReport:
    """
    绩效报告可视化

    生成 IC 曲线、累积收益曲线和风险分析图表。
    支持 matplotlib (静态) 和 plotly (交互) 两种后端。

    使用示例:
        report = PerformanceReport(full_report)
        report.plot_ic_curve().savefig("ic.png")
        report.to_html("full_report.html")
    """

    def __init__(self, full_report: FullReport, backend: str = "matplotlib"):
        self.report = full_report
        self.backend = backend
        self.logger = get_logger()
        self._figures: Dict[str, Any] = {}

    def plot_ic_curve(self, title: str = "IC & Rank IC Time Series"):
        """
        绘制 IC 与 Rank IC 时序曲线

        Returns:
            matplotlib Figure 或 plotly Figure
        """
        ic = self.report.ic_metrics
        if ic is None or len(ic.ic_series) == 0:
            self.logger.warning("无 IC 数据可供绘图")
            return None

        if self.backend == "plotly":
            return self._plot_ic_curve_plotly(ic, title)
        else:
            return self._plot_ic_curve_mpl(ic, title)

    def _plot_ic_curve_mpl(self, ic: ICMetrics, title: str):
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

        # 上图: IC 曲线
        ax1.plot(ic.ic_series.index, ic.ic_series.values,
                 color="#1f77b4", alpha=0.7, linewidth=0.8, label="IC")
        ax1.plot(ic.rank_ic_series.index, ic.rank_ic_series.values,
                 color="#ff7f0e", alpha=0.7, linewidth=0.8, label="Rank IC")
        ax1.axhline(y=0, color="black", linestyle="--", linewidth=0.5)
        ax1.axhline(y=ic.ic_mean, color="#1f77b4", linestyle="--", linewidth=1,
                    label=f"IC Mean={ic.ic_mean:.4f}")
        ax1.axhline(y=ic.rank_ic_mean, color="#ff7f0e", linestyle="--", linewidth=1,
                    label=f"Rank IC Mean={ic.rank_ic_mean:.4f}")
        ax1.set_ylabel("IC Value")
        ax1.set_title(title, fontsize=13, fontweight="bold")
        ax1.legend(loc="upper right", fontsize=8)
        ax1.grid(True, alpha=0.3)

        # 下图: IC 累积和
        ic_cumsum = ic.ic_series.cumsum()
        rank_ic_cumsum = ic.rank_ic_series.cumsum()
        ax2.fill_between(ic_cumsum.index, 0, ic_cumsum.values,
                          color="#1f77b4", alpha=0.3, label="IC CumSum")
        ax2.plot(ic_cumsum.index, ic_cumsum.values,
                 color="#1f77b4", linewidth=1)
        ax2.fill_between(rank_ic_cumsum.index, 0, rank_ic_cumsum.values,
                          color="#ff7f0e", alpha=0.3, label="Rank IC CumSum")
        ax2.plot(rank_ic_cumsum.index, rank_ic_cumsum.values,
                 color="#ff7f0e", linewidth=1)
        ax2.set_ylabel("Cumulative IC")
        ax2.set_xlabel("Date")
        ax2.axhline(y=0, color="black", linestyle="--", linewidth=0.5)
        ax2.legend(loc="upper left", fontsize=8)
        ax2.grid(True, alpha=0.3)

        ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha="right")

        # 添加文本摘要
        text = (
            f"ICIR={ic.icir:.3f} | "
            f"Rank ICIR={ic.rank_icir:.3f} | "
            f"IC>0 Ratio={ic.ic_positive_ratio:.1%} | "
            f"IC Significant={ic.ic_significant_ratio:.1%}"
        )
        fig.text(0.5, 0.01, text, ha="center", fontsize=9,
                 bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

        plt.tight_layout(rect=[0, 0.04, 1, 1])
        self._figures["ic_curve"] = fig
        return fig

    def _plot_ic_curve_plotly(self, ic: ICMetrics, title: str):
        try:
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots
        except ImportError:
            self.logger.warning("plotly 未安装，回退到 matplotlib")
            self.backend = "matplotlib"
            return self._plot_ic_curve_mpl(ic, title)

        fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                            subplot_titles=(title, "Cumulative IC"))

        fig.add_trace(go.Scatter(x=ic.ic_series.index, y=ic.ic_series.values,
                                  mode="lines", name="IC", line=dict(color="#1f77b4")),
                      row=1, col=1)
        fig.add_trace(go.Scatter(x=ic.rank_ic_series.index, y=ic.rank_ic_series.values,
                                  mode="lines", name="Rank IC", line=dict(color="#ff7f0e")),
                      row=1, col=1)

        ic_cumsum = ic.ic_series.cumsum()
        rank_ic_cumsum = ic.rank_ic_series.cumsum()
        fig.add_trace(go.Scatter(x=ic_cumsum.index, y=ic_cumsum.values,
                                  mode="lines", name="IC CumSum",
                                  fill="tozeroy", line=dict(color="#1f77b4")),
                      row=2, col=1)
        fig.add_trace(go.Scatter(x=rank_ic_cumsum.index, y=rank_ic_cumsum.values,
                                  mode="lines", name="Rank IC CumSum",
                                  fill="tozeroy", line=dict(color="#ff7f0e")),
                      row=2, col=1)

        fig.update_layout(height=600, showlegend=True, title_text=title)
        self._figures["ic_curve"] = fig
        return fig

    def plot_cumulative_return(
        self,
        title: str = "Cumulative Return Comparison",
    ):
        """
        绘制累积收益曲线 (含基准对比)

        Returns:
            matplotlib Figure 或 plotly Figure
        """
        rm = self.report.return_metrics
        if rm is None or len(rm.cumulative_returns) == 0:
            self.logger.warning("无收益数据可供绘图")
            return None

        if self.backend == "plotly":
            return self._plot_cumulative_return_plotly(rm, title)
        else:
            return self._plot_cumulative_return_mpl(rm, title)

    def _plot_cumulative_return_mpl(self, rm: ReturnMetrics, title: str):
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates

        fig, ax = plt.subplots(figsize=(14, 6))

        ax.plot(rm.cumulative_returns.index, rm.cumulative_returns.values,
                color="#2ca02c", linewidth=1.5, label="Strategy NAV")

        if rm.excess_returns is not None and len(rm.excess_returns) > 0:
            excess_cum = (1 + rm.excess_returns).cumprod()
            ax2 = ax.twinx()
            ax2.fill_between(excess_cum.index, 1, excess_cum.values,
                              color="#d62728", alpha=0.3, label="Excess Return")
            ax2.set_ylabel("Excess Return", color="#d62728")
            ax2.legend(loc="upper left", fontsize=8)

        ax.axhline(y=1.0, color="black", linestyle="--", linewidth=0.5)
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.set_ylabel("Cumulative Return", color="#2ca02c")
        ax.set_xlabel("Date")
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")

        text = (
            f"Total={rm.total_return:.2%} | "
            f"Ann.Ret={rm.annualized_return:.2%} | "
            f"Sharpe={rm.sharpe_ratio:.2f} | "
            f"MaxDD={rm.max_drawdown:.2%}"
        )
        fig.text(0.5, 0.01, text, ha="center", fontsize=9,
                 bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

        plt.tight_layout(rect=[0, 0.04, 1, 1])
        self._figures["cumulative_return"] = fig
        return fig

    def _plot_cumulative_return_plotly(self, rm: ReturnMetrics, title: str):
        try:
            import plotly.graph_objects as go
        except ImportError:
            self.backend = "matplotlib"
            return self._plot_cumulative_return_mpl(rm, title)

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=rm.cumulative_returns.index, y=rm.cumulative_returns.values,
            mode="lines", name="Strategy NAV", line=dict(color="#2ca02c", width=2)
        ))
        fig.add_hline(y=1.0, line_dash="dash", line_color="gray")
        fig.update_layout(title=title, xaxis_title="Date", yaxis_title="Cumulative Return")
        self._figures["cumulative_return"] = fig
        return fig

    def plot_risk_analysis(
        self,
        title: str = "Risk Analysis Dashboard",
    ):
        """
        绘制风险分析面板 (回撤曲线 + 收益分布 + 滚动指标)

        Returns:
            matplotlib Figure
        """
        rm = self.report.return_metrics
        if rm is None or len(rm.daily_returns) == 0:
            self.logger.warning("无收益数据可供风险分析")
            return None

        rk = self.report.risk_metrics
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates

        fig = plt.figure(figsize=(16, 10))
        gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.3)

        # 1. 回撤曲线 (左上: 占2列)
        ax1 = fig.add_subplot(gs[0, :2])
        cummax = rm.cumulative_returns.cummax()
        drawdowns = (rm.cumulative_returns - cummax) / cummax
        ax1.fill_between(drawdowns.index, 0, drawdowns.values,
                          color="#d62728", alpha=0.5)
        ax1.plot(drawdowns.index, drawdowns.values,
                 color="#d62728", linewidth=0.8)
        ax1.set_title("Drawdown Curve", fontsize=11, fontweight="bold")
        ax1.set_ylabel("Drawdown")
        ax1.axhline(y=0, color="black", linestyle="--", linewidth=0.5)
        ax1.axhline(y=rm.max_drawdown, color="red", linestyle="--", linewidth=1,
                     label=f"MaxDD={rm.max_drawdown:.2%}")
        ax1.legend(loc="lower left", fontsize=8)
        ax1.grid(True, alpha=0.3)

        # 2. 收益分布 (右上)
        ax2 = fig.add_subplot(gs[0, 2])
        rets = rm.daily_returns.dropna()
        ax2.hist(rets.values, bins=50, color="#1f77b4", alpha=0.7, edgecolor="white")
        ax2.axvline(x=0, color="black", linestyle="--", linewidth=1)
        ax2.axvline(x=rets.mean(), color="red", linestyle="--", linewidth=1,
                     label=f"Mean={rets.mean():.4%}")
        ax2.set_title("Daily Return Distribution", fontsize=11, fontweight="bold")
        ax2.set_xlabel("Daily Return")
        ax2.legend(fontsize=8)

        # 3. 滚动 Sharpe (左下: 占2列)
        ax3 = fig.add_subplot(gs[1, :2])
        roll_sharpe = rets.rolling(60).mean() / rets.rolling(60).std() * np.sqrt(252)
        ax3.plot(roll_sharpe.index, roll_sharpe.values, color="#ff7f0e", linewidth=1)
        ax3.axhline(y=0, color="black", linestyle="--", linewidth=0.5)
        ax3.set_title("Rolling Sharpe (60-day)", fontsize=11, fontweight="bold")
        ax3.set_ylabel("Sharpe")
        ax3.grid(True, alpha=0.3)

        # 4. 滚动波动率 (右下)
        ax4 = fig.add_subplot(gs[1, 2])
        roll_vol = rets.rolling(60).std() * np.sqrt(252)
        ax4.plot(roll_vol.index, roll_vol.values, color="#9467bd", linewidth=1)
        ax4.set_title("Rolling Volatility (60-day)", fontsize=11, fontweight="bold")
        ax4.set_ylabel("Volatility")
        ax4.grid(True, alpha=0.3)

        # 5. 风险指标汇总表 (底部)
        ax5 = fig.add_subplot(gs[2, :])
        ax5.axis("off")
        if rk:
            risk_lines = [
                f"VaR(95%): {rk.var_95:.4%}  |  VaR(99%): {rk.var_99:.4%}  |  CVaR(95%): {rk.cvar_95:.4%}",
                f"Max Daily Loss: {rk.max_daily_loss:.4%}  |  Max Daily Gain: {rk.max_daily_gain:.4%}",
                f"Skewness: {rk.skewness:.3f}  |  Kurtosis: {rk.kurtosis:.3f}  |  Stability: {rk.stability:.1%}",
                f"Sharpe: {rm.sharpe_ratio:.2f}  |  Sortino: {rm.sortino_ratio:.2f}  |  Calmar: {rm.calmar_ratio:.2f}",
                f"Win Rate: {rm.win_rate:.1%}  |  P/L Ratio: {rm.profit_loss_ratio:.2f}  |  MaxDD Duration: {rm.max_drawdown_duration}d",
            ]
        else:
            risk_lines = ["无风险指标数据"]
        ax5.text(0.5, 0.5, "\n".join(risk_lines), ha="center", va="center",
                 fontsize=10, transform=ax5.transAxes,
                 bbox=dict(boxstyle="round", facecolor="lightblue", alpha=0.3))

        fig.suptitle(title, fontsize=14, fontweight="bold", y=0.98)
        self._figures["risk_analysis"] = fig
        return fig

    def to_html(
        self,
        path: str,
        include_plots: bool = True,
    ):
        """
        导出完整 HTML 报告

        Args:
            path: 输出文件路径
            include_plots: 是否嵌入图表 (base64)
        """
        html_parts = [
            "<!DOCTYPE html>",
            "<html><head>",
            '<meta charset="utf-8">',
            "<title>Qlib Performance Report</title>",
            "<style>",
            "body { font-family: 'Segoe UI', Arial, sans-serif; max-width: 1200px; margin: 0 auto; padding: 20px; background: #f5f5f5; }",
            "h1 { color: #1f77b4; border-bottom: 3px solid #1f77b4; padding-bottom: 10px; }",
            "h2 { color: #ff7f0e; margin-top: 30px; }",
            "table { border-collapse: collapse; width: 100%%; margin: 15px 0; background: white; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }",
            "th, td { border: 1px solid #ddd; padding: 10px 15px; text-align: left; }",
            "th { background: #1f77b4; color: white; }",
            "tr:nth-child(even) { background: #f9f9f9; }",
            ".metric-good { color: #2ca02c; font-weight: bold; }",
            ".metric-bad { color: #d62728; font-weight: bold; }",
            ".metric-neutral { color: #ff7f0e; }",
            "img { max-width: 100%%; margin: 20px 0; box-shadow: 0 4px 8px rgba(0,0,0,0.15); }",
            ".footer { margin-top: 40px; padding-top: 20px; border-top: 1px solid #ccc; color: #888; font-size: 0.9em; }",
            "</style>",
            "</head><body>",
            f"<h1>📊 Qlib 量化分析绩效报告</h1>",
            f"<p>生成时间: {self.report.timestamp}</p>",
            f"<p>模型: {self.report.model_name or 'N/A'}</p>",
        ]

        # IC 指标表
        if self.report.ic_metrics:
            ic = self.report.ic_metrics
            html_parts += [
                "<h2>📈 信息系数 (IC) 分析</h2>",
                "<table>",
                "<tr><th>指标</th><th>值</th><th>评判</th></tr>",
                f"<tr><td>IC Mean</td><td>{ic.ic_mean:.6f}</td>{self._metric_td(ic.ic_mean, 0.03, 0.05)}</tr>",
                f"<tr><td>IC Std</td><td>{ic.ic_std:.6f}</td><td></td></tr>",
                f"<tr><td>ICIR</td><td>{ic.icir:.4f}</td>{self._metric_td(ic.icir, 0.3, 0.5)}</tr>",
                f"<tr><td>Rank IC Mean</td><td>{ic.rank_ic_mean:.6f}</td>{self._metric_td(ic.rank_ic_mean, 0.03, 0.05)}</tr>",
                f"<tr><td>Rank ICIR</td><td>{ic.rank_icir:.4f}</td>{self._metric_td(ic.rank_icir, 0.3, 0.5)}</tr>",
                f"<tr><td>IC > 0 Ratio</td><td>{ic.ic_positive_ratio:.2%}</td>{self._metric_td(ic.ic_positive_ratio, 0.5, 0.6)}</tr>",
                f"<tr><td>IC Significant Ratio</td><td>{ic.ic_significant_ratio:.2%}</td><td></td></tr>",
                "</table>",
            ]
            if include_plots and "ic_curve" in self._figures:
                img_html = self._figure_to_html_img(self._figures["ic_curve"])
                html_parts.append(img_html)

        # 收益指标表
        if self.report.return_metrics:
            rm = self.report.return_metrics
            html_parts += [
                "<h2>💰 收益与风险分析</h2>",
                "<table>",
                "<tr><th>指标</th><th>值</th><th>评判</th></tr>",
                f"<tr><td>总收益率</td><td>{rm.total_return:.2%}</td>{self._metric_td(rm.total_return, 0.0, 0.1)}</tr>",
                f"<tr><td>年化收益率</td><td>{rm.annualized_return:.2%}</td>{self._metric_td(rm.annualized_return, 0.0, 0.1)}</tr>",
                f"<tr><td>年化波动率</td><td>{rm.annualized_volatility:.2%}</td><td></td></tr>",
                f"<tr><td>Sharpe 比率</td><td>{rm.sharpe_ratio:.2f}</td>{self._metric_td(rm.sharpe_ratio, 0.5, 1.0)}</tr>",
                f"<tr><td>最大回撤</td><td>{rm.max_drawdown:.2%}</td>{self._metric_td(-rm.max_drawdown, -0.15, -0.1)}</tr>",
                f"<tr><td>最大回撤持续(天)</td><td>{rm.max_drawdown_duration}</td><td></td></tr>",
                f"<tr><td>Calmar 比率</td><td>{rm.calmar_ratio:.2f}</td>{self._metric_td(rm.calmar_ratio, 0.5, 1.0)}</tr>",
                f"<tr><td>Sortino 比率</td><td>{rm.sortino_ratio:.2f}</td>{self._metric_td(rm.sortino_ratio, 0.5, 1.0)}</tr>",
                f"<tr><td>胜率</td><td>{rm.win_rate:.2%}</td>{self._metric_td(rm.win_rate, 0.5, 0.55)}</tr>",
                f"<tr><td>盈亏比</td><td>{rm.profit_loss_ratio:.2f}</td>{self._metric_td(rm.profit_loss_ratio, 1.0, 1.5)}</tr>",
                "</table>",
            ]
            if include_plots and "cumulative_return" in self._figures:
                img_html = self._figure_to_html_img(self._figures["cumulative_return"])
                html_parts.append(img_html)
            if include_plots and "risk_analysis" in self._figures:
                img_html = self._figure_to_html_img(self._figures["risk_analysis"])
                html_parts.append(img_html)

        # 风险指标表
        if self.report.risk_metrics:
            rk = self.report.risk_metrics
            html_parts += [
                "<h2>⚠️ 风险度量</h2>",
                "<table>",
                "<tr><th>指标</th><th>值</th></tr>",
                f"<tr><td>VaR (95%)</td><td>{rk.var_95:.4%}</td></tr>",
                f"<tr><td>VaR (99%)</td><td>{rk.var_99:.4%}</td></tr>",
                f"<tr><td>CVaR (95%)</td><td>{rk.cvar_95:.4%}</td></tr>",
                f"<tr><td>最大日亏损</td><td>{rk.max_daily_loss:.4%}</td></tr>",
                f"<tr><td>最大日收益</td><td>{rk.max_daily_gain:.4%}</td></tr>",
                f"<tr><td>偏度 (Skewness)</td><td>{rk.skewness:.4f}</td></tr>",
                f"<tr><td>峰度 (Kurtosis)</td><td>{rk.kurtosis:.4f}</td></tr>",
                f"<tr><td>收益稳定性</td><td>{rk.stability:.2%}</td></tr>",
                "</table>",
            ]

        html_parts += [
            "<div class='footer'>",
            f"<p>Generated by Qlib US Fundamental Analysis System | {self.report.timestamp}</p>",
            "</div>",
            "</body></html>",
        ]

        html = "\n".join(html_parts)
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(html)

        self.logger.info("HTML 报告已生成", path=path)
        return html

    def to_json(self, path: str):
        """导出 JSON 格式报告"""
        data = self.report.to_dict()
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        self.logger.info("JSON 报告已生成", path=path)

    def _metric_td(self, value: float, warn: float, good: float) -> str:
        """根据阈值生成带颜色的表格单元"""
        if value >= good:
            return '<td class="metric-good">✅ 优秀</td>'
        elif value >= warn:
            return '<td class="metric-neutral">⚠️ 一般</td>'
        else:
            return '<td class="metric-bad">❌ 需改进</td>'

    def _figure_to_html_img(self, fig) -> str:
        """将 matplotlib Figure 转为 base64 图片嵌入 HTML"""
        buf = io.BytesIO()
        try:
            fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
        except AttributeError:
            # plotly figure
            fig.write_image(buf, format="png")
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode("utf-8")
        buf.close()
        return f'<img src="data:image/png;base64,{img_base64}" alt="chart">'


# ========================================================================
#  报告导出工具
# ========================================================================

class ReportExporter:
    """
    报告导出器 — 支持多种输出格式

    使用示例:
        exporter = ReportExporter(full_report)
        exporter.export_all("reports/experiment_001")
    """

    def __init__(self, full_report: FullReport, backend: str = "matplotlib"):
        self.report = full_report
        self.performance_report = PerformanceReport(full_report, backend=backend)
        self.logger = get_logger()

    def export_all(self, output_dir: str, prefix: str = "report"):
        """
        导出全部格式: HTML + JSON + 图表PNG

        Args:
            output_dir: 输出目录
            prefix: 文件名前缀
        """
        os.makedirs(output_dir, exist_ok=True)

        # 生成图表
        self.performance_report.plot_ic_curve()
        self.performance_report.plot_cumulative_return()
        self.performance_report.plot_risk_analysis()

        # HTML
        html_path = os.path.join(output_dir, f"{prefix}.html")
        self.performance_report.to_html(html_path)

        # JSON
        json_path = os.path.join(output_dir, f"{prefix}.json")
        self.performance_report.to_json(json_path)

        # PNG (单独保存每张图)
        for name, fig in self.performance_report._figures.items():
            png_path = os.path.join(output_dir, f"{prefix}_{name}.png")
            try:
                fig.savefig(png_path, dpi=150, bbox_inches="tight")
            except AttributeError:
                fig.write_image(png_path)
            self.logger.info("图表已保存", path=png_path)

        self.logger.info("全部报告已导出", dir=output_dir)
        return output_dir
