"""
Alpha 因子库 (Alpha Factor Library)

基于表达式引擎构建的权威财务健康度与量价复合因子集合。
实现国际公认的基本面评估模型与正交化技术。

核心因子:
- Altman Z-Score: 阿特曼破产预警模型 (5因子加权)
- Piotroski F-Score: 皮奥特罗斯基基本面评分 (9项二元测试)
- MACD: 指数平滑异同移动平均线
- Bollinger Bands: 布林带通道
- 正交化: 剥离基本面与量价因子的共线性

使用示例:
    from src.analyzers.alpha_factors import AlphaFactorCalculator
    
    calc = AlphaFactorCalculator()
    factors = calc.compute_all(df)  # 一键计算全部因子
    zscore = calc.altman_zscore(df)  # 单独计算 Z-Score
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

from src.analyzers.expression_engine import ExpressionEngine
from src.utils.logger import get_logger


class AlphaFactorCalculator:
    """
    Alpha 因子计算器

    封装所有基本面和技术面因子的计算逻辑，
    基于 ExpressionEngine 提供向量化加速计算。
    """

    def __init__(self):
        self.engine = ExpressionEngine()
        self.logger = get_logger()

    # ========================================================================
    #  基本面因子
    # ========================================================================

    def altman_zscore(self, df: pd.DataFrame) -> pd.Series:
        """
        Altman Z-Score — 破产预警模型

        原始公式 (制造业):
            Z = 1.2 * X1 + 1.4 * X2 + 3.3 * X3 + 0.6 * X4 + 1.0 * X5
        
        其中:
            X1 = 营运资金 / 总资产      (流动比率)
            X2 = 留存收益 / 总资产      (累计盈利能力)
            X3 = EBIT / 总资产          (资产回报效率)
            X4 = 市值 / 总负债          (偿债能力)
            X5 = 营业收入 / 总资产      (资产周转效率)

        解读:
            Z > 2.99  → 安全区 (Safe Zone)
            1.81 < Z < 2.99 → 灰色区 (Grey Zone)
            Z < 1.81 → 困境区 (Distress Zone)

        Args:
            df: 包含以下列的 DataFrame:
                working_capital, total_assets, retained_earnings,
                ebit, market_cap, total_liabilities, revenue

        Returns:
            Z-Score Series
        """
        required = [
            "working_capital", "total_assets", "retained_earnings",
            "ebit", "market_cap", "total_liabilities", "revenue",
        ]
        self._validate_columns(df, required, "Altman Z-Score")

        # 避免除零
        safe_assets = df["total_assets"].replace(0, np.nan)
        safe_liabilities = df["total_liabilities"].replace(0, np.nan)

        x1 = df["working_capital"] / safe_assets
        x2 = df["retained_earnings"] / safe_assets
        x3 = df["ebit"] / safe_assets
        x4 = df["market_cap"] / safe_liabilities
        x5 = df["revenue"] / safe_assets

        z = 1.2 * x1 + 1.4 * x2 + 3.3 * x3 + 0.6 * x4 + 1.0 * x5

        z.name = "altman_zscore"
        return z

    def altman_zscore_modified(self, df: pd.DataFrame) -> pd.Series:
        """
        修正版 Altman Z-Score (适用于非制造业/新兴市场)

        修正系数:
            Z' = 0.717 * X1 + 0.847 * X2 + 3.107 * X3 + 0.420 * X4 + 0.998 * X5

        X4 使用账面权益替代市值 (book_equity / total_liabilities)

        Args:
            df: 包含 working_capital, total_assets, retained_earnings,
                ebit, book_equity, total_liabilities, revenue

        Returns:
            修正 Z'-Score Series
        """
        required = [
            "working_capital", "total_assets", "retained_earnings",
            "ebit", "book_equity", "total_liabilities", "revenue",
        ]
        self._validate_columns(df, required, "Altman Z'-Score")

        safe_assets = df["total_assets"].replace(0, np.nan)
        safe_liabilities = df["total_liabilities"].replace(0, np.nan)

        x1 = df["working_capital"] / safe_assets
        x2 = df["retained_earnings"] / safe_assets
        x3 = df["ebit"] / safe_assets
        x4 = df["book_equity"] / safe_liabilities
        x5 = df["revenue"] / safe_assets

        z = 0.717 * x1 + 0.847 * x2 + 3.107 * x3 + 0.420 * x4 + 0.998 * x5

        z.name = "altman_zscore_modified"
        return z

    def piotroski_fscore(self, df: pd.DataFrame) -> pd.Series:
        """
        Piotroski F-Score — 基本面健康度评分 (0~9)

        九项二元测试 (每项通过得 1 分):

        【盈利能力 (Profitability) — 4分】
        1. ROA > 0           (净利润/总资产为正)
        2. CFO > 0            (经营现金流为正)
        3. ΔROA > 0          (ROA 同比提升)
        4. CFO > ROA          (现金流质量高于会计利润)

        【财务杠杆与流动性 (Leverage/Liquidity) — 3分】
        5. ΔLongTermDebt < 0  (长期负债下降)
        6. ΔCurrentRatio > 0  (流动比率提升)
        7. No New Shares      (未增发新股)

        【运营效率 (Operating Efficiency) — 2分】
        8. ΔGrossMargin > 0   (毛利率提升)
        9. ΔAssetTurnover > 0 (资产周转率提升)

        解读:
            F >= 7  → 高质量价值股 (High Quality)
            F <= 3  → 低质量股 (Low Quality)

        Args:
            df: 包含以下列的 DataFrame:
                roa, operating_cf, net_income, total_assets,
                long_term_debt, current_ratio, shares_outstanding,
                gross_margin, asset_turnover
                注意: df 应按 instrument 分组并按日期排序

        Returns:
            F-Score Series (0~9)
        """
        required = [
            "roa", "operating_cf", "net_income", "total_assets",
            "long_term_debt", "current_ratio", "shares_outstanding",
            "gross_margin", "asset_turnover",
        ]
        self._validate_columns(df, required, "Piotroski F-Score")

        score = pd.Series(0, index=df.index, name="piotroski_fscore")

        # === 盈利能力 (Profitability) ===

        # 1. ROA > 0
        score += (df["roa"] > 0).astype(int)

        # 2. CFO > 0
        score += (df["operating_cf"] > 0).astype(int)

        # 3. ΔROA > 0 (ROA 同比提升)
        if "instrument" in df.columns:
            delta_roa = df.groupby("instrument")["roa"].diff()
        else:
            delta_roa = df["roa"].diff()
        score += (delta_roa > 0).astype(int)

        # 4. CFO > ROA (应计质量)
        score += (df["operating_cf"] > df["roa"]).astype(int)

        # === 财务杠杆与流动性 (Leverage/Liquidity) ===

        # 5. ΔLongTermDebt < 0 (杠杆率下降 = 好)
        if "instrument" in df.columns:
            delta_ltd = df.groupby("instrument")["long_term_debt"].diff()
        else:
            delta_ltd = df["long_term_debt"].diff()
        score += (delta_ltd < 0).astype(int)

        # 6. ΔCurrentRatio > 0 (流动性提升)
        if "instrument" in df.columns:
            delta_cr = df.groupby("instrument")["current_ratio"].diff()
        else:
            delta_cr = df["current_ratio"].diff()
        score += (delta_cr > 0).astype(int)

        # 7. No New Shares (未增发 = 股本不变或下降)
        if "instrument" in df.columns:
            delta_shares = df.groupby("instrument")["shares_outstanding"].diff()
        else:
            delta_shares = df["shares_outstanding"].diff()
        score += (delta_shares <= 0).astype(int)

        # === 运营效率 (Operating Efficiency) ===

        # 8. ΔGrossMargin > 0 (毛利率提升)
        if "instrument" in df.columns:
            delta_gm = df.groupby("instrument")["gross_margin"].diff()
        else:
            delta_gm = df["gross_margin"].diff()
        score += (delta_gm > 0).astype(int)

        # 9. ΔAssetTurnover > 0 (资产周转率提升)
        if "instrument" in df.columns:
            delta_at = df.groupby("instrument")["asset_turnover"].diff()
        else:
            delta_at = df["asset_turnover"].diff()
        score += (delta_at > 0).astype(int)

        return score

    # ========================================================================
    #  量价技术因子
    # ========================================================================

    def macd(
        self,
        df: pd.DataFrame,
        price_col: str = "close",
        fast: int = 12,
        slow: int = 26,
        signal: int = 9,
    ) -> pd.DataFrame:
        """
        MACD — 指数平滑异同移动平均线

        MACD Line = EMA(fast) - EMA(slow)
        Signal Line = EMA(MACD Line, signal)
        Histogram = MACD Line - Signal Line

        Args:
            df: 包含 price_col 的 DataFrame
            price_col: 价格列名
            fast: 快线周期 (默认 12)
            slow: 慢线周期 (默认 26)
            signal: 信号线周期 (默认 9)

        Returns:
            DataFrame with columns: macd, macd_signal, macd_histogram
        """
        if price_col not in df.columns:
            raise KeyError(f"MACD 需要 '{price_col}' 列，可用列: {list(df.columns)}")

        price = df[price_col]

        ema_fast = price.ewm(span=fast, adjust=False).mean()
        ema_slow = price.ewm(span=slow, adjust=False).mean()

        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        histogram = macd_line - signal_line

        result = pd.DataFrame({
            "macd": macd_line,
            "macd_signal": signal_line,
            "macd_histogram": histogram,
        }, index=df.index)

        return result

    def bollinger_bands(
        self,
        df: pd.DataFrame,
        price_col: str = "close",
        window: int = 20,
        num_std: float = 2.0,
    ) -> pd.DataFrame:
        """
        Bollinger Bands — 布林带通道

        Middle Band = MA(window)
        Upper Band  = Middle + num_std * Std(window)
        Lower Band  = Middle - num_std * Std(window)

        附加指标:
            %b = (price - lower) / (upper - lower)     — 价格在带内的相对位置
            bandwidth = (upper - lower) / middle         — 带宽 (波动率)

        Args:
            df: 包含 price_col 的 DataFrame
            price_col: 价格列名
            window: 移动平均窗口 (默认 20)
            num_std: 标准差倍数 (默认 2.0)

        Returns:
            DataFrame: bb_middle, bb_upper, bb_lower, bb_pct_b, bb_bandwidth
        """
        if price_col not in df.columns:
            raise KeyError(f"Bollinger Bands 需要 '{price_col}' 列")

        price = df[price_col]

        middle = price.rolling(window=window, min_periods=1).mean()
        std = price.rolling(window=window, min_periods=1).std()

        upper = middle + num_std * std
        lower = middle - num_std * std

        # %b — 价格在带内的相对位置 [0, 1]
        band_range = upper - lower
        pct_b = (price - lower) / band_range.replace(0, np.nan)

        # Bandwidth — 带宽百分比
        bandwidth = (upper - lower) / middle.replace(0, np.nan)

        result = pd.DataFrame({
            "bb_middle": middle,
            "bb_upper": upper,
            "bb_lower": lower,
            "bb_pct_b": pct_b,
            "bb_bandwidth": bandwidth,
        }, index=df.index)

        return result

    def rsi(
        self,
        df: pd.DataFrame,
        price_col: str = "close",
        window: int = 14,
    ) -> pd.Series:
        """
        RSI — 相对强弱指标

        RSI = 100 - (100 / (1 + RS))
        RS = Average Gain / Average Loss (过去 window 期)

        Args:
            df: 包含 price_col 的 DataFrame
            price_col: 价格列名
            window: 计算周期 (默认 14)

        Returns:
            RSI Series (0~100)
        """
        if price_col not in df.columns:
            raise KeyError(f"RSI 需要 '{price_col}' 列")

        price = df[price_col]
        delta = price.diff()

        gain = delta.clip(lower=0)
        loss = (-delta).clip(lower=0)

        avg_gain = gain.ewm(alpha=1/window, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1/window, adjust=False).mean()

        rs = avg_gain / avg_loss.replace(0, np.nan)
        rsi = 100.0 - (100.0 / (1.0 + rs))

        rsi.name = "rsi"
        return rsi

    # ========================================================================
    #  因子正交化
    # ========================================================================

    def orthogonalize(
        self,
        target: pd.Series,
        reference: pd.Series,
        method: str = "regression",
    ) -> pd.Series:
        """
        因子正交化 — 剥离 target 中由 reference 解释的部分

        通过线性回归将 target 投影到 reference 的正交补空间:
            residual = target - β * reference
        其中 β = Cov(target, reference) / Var(reference)

        Args:
            target: 待正交化的因子 (如基本面因子)
            reference: 参考因子 (如动量因子)
            method: 'regression' | 'difference' | 'ratio'

        Returns:
            正交化后的残差 Series
        """
        valid_mask = target.notna() & reference.notna()

        if valid_mask.sum() < 10:
            self.logger.warning("正交化样本不足", n_valid=valid_mask.sum())
            return target

        t = target[valid_mask].values
        r = reference[valid_mask].values

        if method == "regression":
            # OLS: residual = target - beta * reference
            beta = np.cov(t, r)[0, 1] / (np.var(r) + 1e-12)
            residual = target.copy()
            residual[valid_mask] = t - beta * r
            residual.name = f"{target.name or 'target'}_orth"
            return residual

        elif method == "difference":
            # 简单差分: 先标准化后相减
            t_std = (t - np.nanmean(t)) / (np.nanstd(t) + 1e-12)
            r_std = (r - np.nanmean(r)) / (np.nanstd(r) + 1e-12)
            residual = target.copy()
            residual[valid_mask] = t_std - r_std
            residual.name = f"{target.name or 'target'}_diff"
            return residual

        elif method == "ratio":
            # 比率正交
            residual = target.copy()
            residual[valid_mask] = t / (r + 1e-12)
            residual.name = f"{target.name or 'target'}_ratio"
            return residual

        else:
            raise ValueError(f"未知正交化方法: {method}")

    def orthogonalize_multi(
        self,
        target: pd.Series,
        references: pd.DataFrame,
    ) -> pd.Series:
        """
        多元正交化 — 同时剥离多个参考因子的共线性

        使用多元线性回归:
            residual = target - X @ β
        其中 X = references (列向量矩阵), β = (X'X)^(-1) X'y

        Args:
            target: 待正交化的因子
            references: 参考因子矩阵 (每列一个因子)

        Returns:
            正交化残差 Series
        """
        valid_mask = target.notna() & references.notna().all(axis=1)

        if valid_mask.sum() < max(10, len(references.columns) * 5):
            self.logger.warning("多元正交化样本不足", n_valid=valid_mask.sum())
            return target

        y = target[valid_mask].values
        X = references[valid_mask].values

        # 添加截距项
        X = np.column_stack([np.ones(len(X)), X])

        try:
            # 正规方程: β = (X'X)^(-1) X'y
            beta = np.linalg.lstsq(X, y, rcond=None)[0]
            y_pred = X @ beta
            residual_vals = y - y_pred
        except np.linalg.LinAlgError:
            self.logger.warning("多元正交化矩阵奇异，回退到简单差分")
            return self.orthogonalize(target, references.iloc[:, 0], method="regression")

        residual = target.copy()
        residual[valid_mask] = residual_vals
        residual.name = f"{target.name or 'target'}_multi_orth"
        return residual

    # ========================================================================
    #  综合计算
    # ========================================================================

    def compute_all(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        一键计算全部可用因子

        自动检测 DataFrame 中可用的列，计算相应的因子集合。

        Args:
            df: 特征 DataFrame

        Returns:
            包含所有因子的 DataFrame
        """
        factors: Dict[str, pd.Series] = {}
        cols = set(df.columns)

        # --- Altman Z-Score ---
        zscore_cols = {"working_capital", "total_assets", "retained_earnings",
                        "ebit", "market_cap", "total_liabilities", "revenue"}
        if zscore_cols.issubset(cols):
            factors["altman_zscore"] = self.altman_zscore(df)
            self.logger.info("计算 Altman Z-Score 完成")

        # --- Altman Z'-Score (非制造业) ---
        zscorem_cols = {"working_capital", "total_assets", "retained_earnings",
                         "ebit", "book_equity", "total_liabilities", "revenue"}
        if zscorem_cols.issubset(cols):
            factors["altman_zscore_modified"] = self.altman_zscore_modified(df)

        # --- Piotroski F-Score ---
        fscore_cols = {"roa", "operating_cf", "net_income", "total_assets",
                       "long_term_debt", "current_ratio", "shares_outstanding",
                       "gross_margin", "asset_turnover"}
        if fscore_cols.issubset(cols):
            factors["piotroski_fscore"] = self.piotroski_fscore(df)
            self.logger.info("计算 Piotroski F-Score 完成")

        # --- MACD ---
        if "close" in cols:
            macd_df = self.macd(df)
            for c in macd_df.columns:
                factors[c] = macd_df[c]
            self.logger.info("计算 MACD 完成")

        # --- Bollinger Bands ---
        if "close" in cols:
            bb_df = self.bollinger_bands(df)
            for c in bb_df.columns:
                factors[c] = bb_df[c]
            self.logger.info("计算 Bollinger Bands 完成")

        # --- RSI ---
        if "close" in cols:
            factors["rsi"] = self.rsi(df)
            self.logger.info("计算 RSI 完成")

        # --- 衍生因子 ---
        if "macd_histogram" in factors and "close" in cols:
            # MACD 动量背离
            factors["macd_divergence"] = (
                factors["macd_histogram"] - factors["macd_histogram"].shift(1)
            )

        if "bb_pct_b" in factors and "close" in cols:
            # 布林带挤压 (squeeze) — 带宽缩小 = 即将突破
            bb_bandwidth = factors.get("bb_bandwidth")
            if bb_bandwidth is not None:
                factors["bb_squeeze"] = (
                    bb_bandwidth < bb_bandwidth.rolling(20).quantile(0.2)
                ).astype(float)

        # --- 复合质量因子 ---
        if "altman_zscore" in factors and "piotroski_fscore" in factors:
            # 综合质量得分 (Z-Score 标准化 + F-Score 标准化 的等权平均)
            z_norm = self._robust_normalize(factors["altman_zscore"])
            f_norm = self._robust_normalize(factors["piotroski_fscore"].astype(float))
            factors["quality_composite"] = (z_norm + f_norm) / 2.0

        return pd.DataFrame(factors, index=df.index)

    # ========================================================================
    #  辅助方法
    # ========================================================================

    def _validate_columns(self, df: pd.DataFrame, required: List[str], factor_name: str):
        """验证 DataFrame 是否包含必需列"""
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise KeyError(
                f"{factor_name} 缺少列: {missing}\n"
                f"可用列: {list(df.columns)}"
            )

    @staticmethod
    def _robust_normalize(series: pd.Series) -> pd.Series:
        """稳健标准化 (中位数 + MAD)"""
        median = series.median()
        mad = (series - median).abs().median()
        if mad == 0:
            mad = series.std()
        if mad == 0 or pd.isna(mad):
            return pd.Series(0, index=series.index)
        return (series - median) / (1.4826 * mad)
