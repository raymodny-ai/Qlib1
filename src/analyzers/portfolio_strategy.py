"""
组合优化与回测策略 (Portfolio Optimization & Backtest)

基于 Qlib TopkDropoutStrategy 理念实现的资产配置与回测引擎。
将模型预测得分转化为可执行的投资组合权重和调仓指令。

核心组件:
- TopkDropoutStrategy: 基于预测排序的动量淘汰策略
- EqualWeightStrategy: 等权配置策略
- ScoreWeightStrategy: 预测得分加权策略
- PortfolioSimulator: 回测模拟器 (含交易成本)
- RiskManager: 风控模块 (最大回撤/止损/波动率目标)

设计原则:
- 严格防过拟合 (训练/回测时间隔离)
- 真实交易成本建模 (佣金 + 滑点)
- 可配置的风控约束
- 完整的回测绩效指标计算

使用示例:
    from src.analyzers.portfolio_strategy import TopkDropoutStrategy, PortfolioSimulator
    
    strategy = TopkDropoutStrategy(top_k=30, dropout_threshold=0.2)
    simulator = PortfolioSimulator(strategy, initial_capital=1_000_000)
    result = simulator.run(predictions_df, price_df)
    print(result.summary())
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np
import pandas as pd

from src.utils.logger import get_logger


# ========================================================================
#  枚举与配置
# ========================================================================

class WeightMethod(str, Enum):
    """权重分配方法"""
    EQUAL = "equal"              # 等权
    SCORE = "score"              # 得分加权
    SCORE_SQRT = "score_sqrt"    # 得分平方根加权
    RANK = "rank"               # 排名加权
    INV_VOL = "inv_vol"          # 逆波动率加权


class SignalType(str, Enum):
    """信号方向"""
    LONG = "long"
    SHORT = "short"


@dataclass
class StrategyConfig:
    """策略配置"""
    top_k: int = 30                      # 持仓股票数
    min_k: int = 10                      # 最少持仓数
    dropout_threshold: float = 0.2       # 淘汰阈值 (排名后 N% 卖出)
    rebalance_freq: int = 1              # 调仓频率 (交易日)
    weight_method: WeightMethod = WeightMethod.EQUAL
    turnover_limit: float = 0.5         # 单日换手率上限
    max_weight_per_stock: float = 0.1   # 单票最大权重
    commission_rate: float = 0.001      # 佣金费率
    slippage_bps: float = 1.0           # 滑点 (基点)
    max_drawdown_limit: float = 0.15    # 最大回撤熔断线
    stop_loss: float = 0.08             # 单票止损线


# ========================================================================
#  回测数据结构
# ========================================================================

@dataclass
class TradeRecord:
    """交易记录"""
    date: str
    instrument: str
    action: str    # "buy" | "sell"
    quantity: int
    price: float
    commission: float
    slippage_cost: float
    notional: float


@dataclass
class Position:
    """持仓记录"""
    instrument: str
    quantity: int
    avg_cost: float
    current_price: float
    market_value: float
    weight: float
    pnl: float
    holding_days: int = 0


@dataclass
class BacktestResult:
    """回测结果"""
    initial_capital: float
    final_capital: float
    total_return: float
    annualized_return: float
    annualized_volatility: float
    sharpe_ratio: float
    max_drawdown: float
    max_drawdown_duration: int
    win_rate: float
    profit_loss_ratio: float
    total_trades: int
    turnover_rate: float
    total_commission: float
    total_slippage: float
    daily_returns: pd.Series
    nav_curve: pd.Series
    benchmark_nav: Optional[pd.Series] = None
    positions_history: List[Dict[str, Any]] = field(default_factory=list)

    def summary(self) -> str:
        """生成回测摘要"""
        lines = [
            "=" * 55,
            "  回测绩效报告 (Backtest Performance Report)",
            "=" * 55,
            f"  初始资金:          ${self.initial_capital:,.0f}",
            f"  最终资金:          ${self.final_capital:,.0f}",
            f"  总收益率:          {self.total_return:.2%}",
            f"  年化收益率:        {self.annualized_return:.2%}",
            f"  年化波动率:        {self.annualized_volatility:.2%}",
            f"  夏普比率:           {self.sharpe_ratio:.2f}",
            f"  最大回撤:           {self.max_drawdown:.2%}",
            f"  最大回撤持续:      {self.max_drawdown_duration} 天",
            f"  胜率:               {self.win_rate:.2%}",
            f"  盈亏比:             {self.profit_loss_ratio:.2f}",
            f"  总交易次数:        {self.total_trades}",
            f"  换手率 (日均):     {self.turnover_rate:.2%}",
            f"  总佣金:            ${self.total_commission:,.0f}",
            f"  总滑点成本:        ${self.total_slippage:,.0f}",
            "=" * 55,
        ]
        return "\n".join(lines)


# ========================================================================
#  权重分配器
# ========================================================================

class WeightAllocator:
    """权重分配器 — 根据得分计算组合权重"""

    def __init__(
        self,
        method: WeightMethod = WeightMethod.EQUAL,
        max_weight: float = 0.1,
    ):
        self.method = method
        self.max_weight = max_weight

    def allocate(
        self,
        scores: pd.Series,
        n_stocks: Optional[int] = None,
    ) -> pd.Series:
        """
        根据得分分配权重

        Args:
            scores: 股票得分 Series (index = instrument)
            n_stocks: 目标持仓数 (None = 全部)

        Returns:
            权重 Series (总和为 1.0)
        """
        if len(scores) == 0:
            return pd.Series(dtype=float)

        # 取 top N
        if n_stocks and n_stocks < len(scores):
            scores = scores.nlargest(n_stocks)

        if self.method == WeightMethod.EQUAL:
            weights = pd.Series(1.0 / len(scores), index=scores.index)

        elif self.method == WeightMethod.SCORE:
            # 得分加权 (负分归零)
            pos = scores.clip(lower=0)
            total = pos.sum()
            if total > 0:
                weights = pos / total
            else:
                weights = pd.Series(1.0 / len(scores), index=scores.index)

        elif self.method == WeightMethod.SCORE_SQRT:
            pos = scores.clip(lower=0)
            sqrt_scores = np.sqrt(pos)
            total = sqrt_scores.sum()
            if total > 0:
                weights = sqrt_scores / total
            else:
                weights = pd.Series(1.0 / len(scores), index=scores.index)

        elif self.method == WeightMethod.RANK:
            ranks = scores.rank(ascending=True)
            total = ranks.sum()
            if total > 0:
                weights = ranks / total
            else:
                weights = pd.Series(1.0 / len(scores), index=scores.index)

        elif self.method == WeightMethod.INV_VOL:
            weights = pd.Series(1.0 / len(scores), index=scores.index)
            self.logger = get_logger()
            self.logger.warning("INV_VOL 需要波动率数据，回退到等权")

        else:
            weights = pd.Series(1.0 / len(scores), index=scores.index)

        # 单票上限约束
        weights = weights.clip(upper=self.max_weight)
        weights = weights / weights.sum()  # 重新归一化

        return weights


# ========================================================================
#  风控模块
# ========================================================================

class RiskManager:
    """
    风控模块

    实时监控组合风险指标，触发熔断/止损机制。
    """

    def __init__(
        self,
        max_drawdown: float = 0.15,
        stop_loss: float = 0.08,
    ):
        self.max_drawdown = max_drawdown
        self.stop_loss = stop_loss
        self._peak_nav = 1.0
        self._current_drawdown = 0.0
        self._stopped_out: Set[str] = set()

    def check_drawdown_breach(self, current_nav: float) -> bool:
        """
        检查是否突破最大回撤线

        Returns:
            True = 触发熔断
        """
        self._peak_nav = max(self._peak_nav, current_nav)
        self._current_drawdown = (self._peak_nav - current_nav) / self._peak_nav
        return self._current_drawdown >= self.max_drawdown

    def check_stop_loss(
        self,
        instrument: str,
        entry_price: float,
        current_price: float,
    ) -> bool:
        """
        检查单票是否触发止损

        Returns:
            True = 触发止损
        """
        if instrument in self._stopped_out:
            return True

        loss = (current_price - entry_price) / entry_price
        if loss <= -self.stop_loss:
            self._stopped_out.add(instrument)
            return True

        return False

    @property
    def current_drawdown(self) -> float:
        return self._current_drawdown

    @property
    def stopped_out_instruments(self) -> Set[str]:
        return self._stopped_out.copy()

    def reset(self):
        """重置风控状态"""
        self._peak_nav = 1.0
        self._current_drawdown = 0.0
        self._stopped_out.clear()


# ========================================================================
#  组合策略 — 抽象基类
# ========================================================================

class BaseStrategy(ABC):
    """组合策略抽象基类"""

    def __init__(self, config: StrategyConfig):
        self.config = config
        self.allocator = WeightAllocator(
            method=config.weight_method,
            max_weight=config.max_weight_per_stock,
        )
        self.risk_manager = RiskManager(
            max_drawdown=config.max_drawdown_limit,
            stop_loss=config.stop_loss,
        )
        self.logger = get_logger()

    @abstractmethod
    def generate_weights(
        self,
        scores: pd.Series,
        current_positions: Dict[str, Position],
        prices: pd.Series,
        date: str,
    ) -> pd.Series:
        """
        根据当前得分和持仓生成目标权重

        Args:
            scores: 当前截面预测得分 (index = instrument)
            current_positions: 当前持仓 {instrument: Position}
            prices: 当前价格
            date: 当前日期

        Returns:
            目标权重 Series (index = instrument)
        """
        ...

    @abstractmethod
    def should_rebalance(self, date: str, day_count: int) -> bool:
        """判断当前是否应调仓"""
        ...

    def calculate_turnover(
        self,
        target_weights: pd.Series,
        current_weights: pd.Series,
    ) -> float:
        """计算换手率"""
        all_instruments = target_weights.index.union(current_weights.index)
        tw = target_weights.reindex(all_instruments, fill_value=0)
        cw = current_weights.reindex(all_instruments, fill_value=0)
        return (tw - cw).abs().sum() / 2.0

    def apply_transaction_cost(
        self,
        trade_value: float,
        is_buy: bool,
    ) -> Tuple[float, float]:
        """
        计算交易成本

        Returns:
            (commission, slippage_cost)
        """
        commission = trade_value * self.config.commission_rate
        slippage = trade_value * (self.config.slippage_bps / 10000)
        return commission, slippage


# ========================================================================
#  TopkDropoutStrategy — 动量淘汰策略
# ========================================================================

class TopkDropoutStrategy(BaseStrategy):
    """
    TopkDropout 动量淘汰策略

    策略逻辑:
    1. 每期选出预测得分前 top_k 的股票
    2. 对已持有但排名跌出阈值 (如后 20%) 的股票强制卖出 (Dropout)
    3. 将腾出的资金按目标权重分配给新入选的股票
    4. 单票最大权重约束 + 换手率限制

    这是 Qlib 中最经典的基本面量化配置策略，适合与
    LightGBM/XGBoost 预测模型配合使用。
    """

    def __init__(self, config: Optional[StrategyConfig] = None, **kwargs):
        if config is None:
            config = StrategyConfig(**kwargs)
        super().__init__(config)

    def generate_weights(
        self,
        scores: pd.Series,
        current_positions: Dict[str, Position],
        prices: pd.Series,
        date: str,
    ) -> pd.Series:
        """
        生成目标权重

        核心流程:
        1. 按得分排序，选出 top_k 候选池
        2. 标记当前持仓中的淘汰标的
        3. 生成新的等权/得分加权分配
        """
        if len(scores) == 0:
            return pd.Series(dtype=float)

        # 按得分降序
        sorted_scores = scores.sort_values(ascending=False)

        # 淘汰逻辑: 当前持仓中排名低于 top_k * (1 + dropout_threshold) 的卖出
        dropout_rank = max(
            self.config.top_k,
            int(self.config.top_k * (1 + self.config.dropout_threshold)),
        )

        # 持仓中需要保留的 (在前 dropout_rank 内)
        held_instruments = set(current_positions.keys())
        keep_pool = set(sorted_scores.iloc[:dropout_rank].index)
        sell_instruments = held_instruments - keep_pool

        self.logger.debug(
            "调仓决策",
            date=date,
            hold=len(held_instruments),
            keep=len(held_instruments & keep_pool),
            sell=len(sell_instruments),
            buy_pool=min(self.config.top_k, len(sorted_scores)),
        )

        # 构建目标持仓: top_k 最优标的
        target_instruments = sorted_scores.iloc[:self.config.top_k].index.tolist()

        # 分配权重
        target_scores = sorted_scores[target_instruments]
        weights = self.allocator.allocate(target_scores)

        return weights

    def should_rebalance(self, date: str, day_count: int) -> bool:
        return day_count % self.config.rebalance_freq == 0


# ========================================================================
#  EqualWeightStrategy — 等权策略
# ========================================================================

class EqualWeightStrategy(BaseStrategy):
    """
    等权配置策略

    对 top_k 股票平均分配权重，适用于基准对比。
    """

    def __init__(self, config: Optional[StrategyConfig] = None, **kwargs):
        if config is None:
            config = StrategyConfig(**kwargs)
        super().__init__(config)

    def generate_weights(
        self,
        scores: pd.Series,
        current_positions: Dict[str, Position],
        prices: pd.Series,
        date: str,
    ) -> pd.Series:
        if len(scores) == 0:
            return pd.Series(dtype=float)

        sorted_scores = scores.sort_values(ascending=False)
        top_n = sorted_scores.iloc[:self.config.top_k]
        n = len(top_n)
        weights = pd.Series(1.0 / n, index=top_n.index)
        return weights.clip(upper=self.config.max_weight_per_stock)

    def should_rebalance(self, date: str, day_count: int) -> bool:
        return day_count % self.config.rebalance_freq == 0


# ========================================================================
#  ScoreWeightStrategy — 得分加权策略
# ========================================================================

class ScoreWeightStrategy(BaseStrategy):
    """
    预测得分加权策略

    直接按模型预测得分分配权重，高分 = 高权重。
    """

    def __init__(self, config: Optional[StrategyConfig] = None, **kwargs):
        if config is None:
            # 得分加权默认参数
            kwargs.setdefault("weight_method", WeightMethod.SCORE)
            config = StrategyConfig(**kwargs)
        super().__init__(config)

    def generate_weights(
        self,
        scores: pd.Series,
        current_positions: Dict[str, Position],
        prices: pd.Series,
        date: str,
    ) -> pd.Series:
        if len(scores) == 0:
            return pd.Series(dtype=float)

        sorted_scores = scores.sort_values(ascending=False)
        top_scores = sorted_scores.iloc[:self.config.top_k]
        return self.allocator.allocate(top_scores)

    def should_rebalance(self, date: str, day_count: int) -> bool:
        return day_count % self.config.rebalance_freq == 0


# ========================================================================
#  PortfolioSimulator — 回测模拟器
# ========================================================================

class PortfolioSimulator:
    """
    组合回测模拟器

    模拟真实交易环境，包含:
    - 严格的交易成本 (佣金 + 滑点)
    - 熔断/止损风控
    - 完整的绩效指标统计
    - NAV 曲线计算

    使用示例:
        simulator = PortfolioSimulator(strategy, initial_capital=1_000_000)
        result = simulator.run(predictions_df, price_df)
        print(result.summary())
    """

    def __init__(
        self,
        strategy: BaseStrategy,
        initial_capital: float = 1_000_000.0,
        benchmark_prices: Optional[pd.Series] = None,
    ):
        self.strategy = strategy
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.benchmark_prices = benchmark_prices
        self.logger = get_logger()

        # 状态
        self.positions: Dict[str, Position] = {}
        self.trade_history: List[TradeRecord] = []
        self.nav_history: List[Tuple[str, float]] = []
        self.position_history: List[Dict[str, Any]] = []

        # 统计
        self._day_count = 0
        self._haulted = False  # 熔断标志

    def run(
        self,
        predictions: pd.DataFrame,
        prices: pd.DataFrame,
        verbose: bool = True,
    ) -> BacktestResult:
        """
        执行回测

        Args:
            predictions: 预测得分 DataFrame
                - index: 日期
                - columns: instrument (股票代码)
                - values: 预测得分
            prices: 价格 DataFrame (同结构)

        Returns:
            BacktestResult
        """
        self._reset()

        dates = sorted(predictions.index)
        n_days = len(dates)

        for i, date in enumerate(dates):
            if isinstance(date, pd.Timestamp):
                date_str = date.strftime("%Y-%m-%d")
            else:
                date_str = str(date)

            # 获取当日预测和价格
            day_scores = predictions.loc[date].dropna()
            day_prices = prices.loc[date].dropna() if date in prices.index else pd.Series()

            # 更新持仓市值
            self._mark_to_market(day_prices, date_str)

            # 检查是否调仓
            if self.strategy.should_rebalance(date_str, self._day_count):
                self._rebalance(day_scores, day_prices, date_str)

            # 风控检查
            nav = self._calculate_nav()
            self.nav_history.append((date_str, nav))

            if self.strategy.risk_manager.check_drawdown_breach(nav):
                self._haulted = True
                self.logger.warning(
                    "触发最大回撤熔断!",
                    date=date_str,
                    drawdown=f"{self.strategy.risk_manager.current_drawdown:.2%}",
                )

            self._day_count += 1

            if verbose and (i % max(1, n_days // 10) == 0):
                self.logger.info(
                    f"回测进度: {i+1}/{n_days}",
                    date=date_str,
                    nav=f"${nav:,.0f}",
                    positions=len(self.positions),
                )

        # 最终清仓
        if self.positions:
            final_prices = prices.iloc[-1].dropna() if len(prices) > 0 else pd.Series()
            self._liquidate(final_prices, dates[-1])

        return self._build_result()

    def _rebalance(
        self,
        scores: pd.Series,
        prices: pd.Series,
        date: str,
    ):
        """执行一次调仓"""
        # 生成目标权重
        target_weights = self.strategy.generate_weights(
            scores, self.positions, prices, date
        )

        if len(target_weights) == 0:
            return

        # 当前权重
        current_nav = self._calculate_nav()
        current_weights = pd.Series({
            inst: pos.market_value / current_nav
            for inst, pos in self.positions.items()
        }) if current_nav > 0 else pd.Series(dtype=float)

        # 换手率限制
        turnover = self.strategy.calculate_turnover(target_weights, current_weights)
        if turnover > self.strategy.config.turnover_limit:
            # 缩减调仓幅度
            scale = self.strategy.config.turnover_limit / turnover
            target_weights = current_weights + scale * (target_weights - current_weights)
            target_weights = target_weights.clip(lower=0)
            target_weights = target_weights / target_weights.sum()

        # === 卖出 ===
        for inst in list(self.positions.keys()):
            if inst not in target_weights.index:
                self._sell_position(inst, prices.get(inst, 0), date)

        # === 买入/调整 ===
        for inst, target_w in target_weights.items():
            if target_w <= 0:
                continue

            price = prices.get(inst, np.nan)
            if pd.isna(price) or price <= 0:
                continue

            target_value = current_nav * target_w

            if inst in self.positions:
                current_value = self.positions[inst].market_value
                diff = target_value - current_value
                if diff > 0:
                    self._buy_position(inst, price, diff, date)
                elif diff < 0:
                    self._sell_partial(inst, price, abs(diff), date)
            else:
                self._buy_position(inst, price, target_value, date)

        # 记录持仓
        self.position_history.append({
            "date": date,
            "nav": current_nav,
            "n_positions": len(self.positions),
            "instruments": list(self.positions.keys()),
        })

    def _buy_position(
        self,
        instrument: str,
        price: float,
        value: float,
        date: str,
    ):
        """买入"""
        commission, slippage = self.strategy.apply_transaction_cost(value, is_buy=True)
        net_value = value - commission - slippage
        if pd.isna(price) or pd.isna(net_value) or price <= 0 or net_value <= 0:
            return
        quantity = int(net_value / price)

        if quantity <= 0:
            return

        total_cost = quantity * price + commission + slippage
        self.capital -= total_cost

        if instrument in self.positions:
            pos = self.positions[instrument]
            total_qty = pos.quantity + quantity
            total_cost_basis = pos.avg_cost * pos.quantity + quantity * price + commission + slippage
            pos.quantity = total_qty
            pos.avg_cost = total_cost_basis / total_qty if total_qty > 0 else 0
            pos.current_price = price
            pos.market_value = total_qty * price
        else:
            self.positions[instrument] = Position(
                instrument=instrument,
                quantity=quantity,
                avg_cost=(quantity * price + commission + slippage) / quantity,
                current_price=price,
                market_value=quantity * price,
                weight=0,
                pnl=0,
            )

        self.trade_history.append(TradeRecord(
            date=date, instrument=instrument, action="buy",
            quantity=quantity, price=price,
            commission=commission, slippage_cost=slippage,
            notional=value,
        ))

    def _sell_position(self, instrument: str, price: float, date: str):
        """卖出全部"""
        if instrument not in self.positions:
            return
        pos = self.positions.pop(instrument)
        value = pos.quantity * price
        commission, slippage = self.strategy.apply_transaction_cost(value, is_buy=False)
        self.capital += value - commission - slippage

        self.trade_history.append(TradeRecord(
            date=date, instrument=instrument, action="sell",
            quantity=pos.quantity, price=price,
            commission=commission, slippage_cost=slippage,
            notional=value,
        ))

    def _sell_partial(self, instrument: str, price: float, value: float, date: str):
        """卖出部分"""
        if instrument not in self.positions:
            return
        pos = self.positions[instrument]
        sell_quantity = min(int(value / price), pos.quantity) if price > 0 else pos.quantity

        if sell_quantity <= 0:
            return

        sell_value = sell_quantity * price
        commission, slippage = self.strategy.apply_transaction_cost(sell_value, is_buy=False)
        self.capital += sell_value - commission - slippage

        pos.quantity -= sell_quantity
        if pos.quantity <= 0:
            self.positions.pop(instrument)
        else:
            pos.market_value = pos.quantity * price

        self.trade_history.append(TradeRecord(
            date=date, instrument=instrument, action="sell",
            quantity=sell_quantity, price=price,
            commission=commission, slippage_cost=slippage,
            notional=sell_value,
        ))

    def _liquidate(self, prices: pd.Series, date):
        """清仓所有持仓"""
        for inst in list(self.positions.keys()):
            price = prices.get(inst, 0) if len(prices) > 0 else 0
            if price > 0:
                self._sell_position(inst, price, str(date))

    def _mark_to_market(self, prices: pd.Series, date: str):
        """按市价更新持仓"""
        for inst, pos in self.positions.items():
            if inst in prices.index and prices[inst] > 0:
                pos.current_price = prices[inst]
                pos.market_value = pos.quantity * pos.current_price
                pos.pnl = pos.market_value - pos.quantity * pos.avg_cost
                pos.holding_days += 1

    def _calculate_nav(self) -> float:
        """计算当前净值"""
        positions_value = sum(p.market_value for p in self.positions.values())
        return self.capital + positions_value

    def _reset(self):
        """重置模拟器状态"""
        self.capital = self.initial_capital
        self.positions.clear()
        self.trade_history.clear()
        self.nav_history.clear()
        self.position_history.clear()
        self._day_count = 0
        self._haulted = False
        self.strategy.risk_manager.reset()

    def _build_result(self) -> BacktestResult:
        """构建回测结果"""
        nav_df = pd.DataFrame(self.nav_history, columns=["date", "nav"])
        nav_df["date"] = pd.to_datetime(nav_df["date"])
        nav_df = nav_df.set_index("date")
        nav_curve = nav_df["nav"]

        # 日收益率
        daily_returns = nav_curve.pct_change().dropna()

        if len(daily_returns) < 2:
            return BacktestResult(
                initial_capital=self.initial_capital,
                final_capital=nav_curve.iloc[-1] if len(nav_curve) > 0 else self.initial_capital,
                total_return=0,
                annualized_return=0,
                annualized_volatility=0,
                sharpe_ratio=0,
                max_drawdown=0,
                max_drawdown_duration=0,
                win_rate=0,
                profit_loss_ratio=0,
                total_trades=0,
                turnover_rate=0,
                total_commission=0,
                total_slippage=0,
                daily_returns=daily_returns,
                nav_curve=nav_curve,
                positions_history=self.position_history,
            )

        # 核心指标
        total_return = nav_curve.iloc[-1] / self.initial_capital - 1
        n_years = len(daily_returns) / 252
        annualized_return = (1 + total_return) ** (1 / max(n_years, 0.01)) - 1
        annualized_vol = daily_returns.std() * np.sqrt(252)
        sharpe = (daily_returns.mean() / daily_returns.std() * np.sqrt(252)) if daily_returns.std() > 0 else 0

        # 最大回撤
        cummax = nav_curve.cummax()
        drawdowns = (nav_curve - cummax) / cummax
        max_dd = drawdowns.min()

        # 最大回撤持续期
        dd_start = None
        max_dd_duration = 0
        current_duration = 0
        for i, dd in enumerate(drawdowns):
            if dd < 0:
                if dd_start is None:
                    dd_start = i
                current_duration = i - dd_start + 1
                max_dd_duration = max(max_dd_duration, current_duration)
            else:
                dd_start = None
                current_duration = 0

        # 胜率与盈亏比
        wins = (daily_returns > 0).sum()
        total = len(daily_returns)
        win_rate = wins / total if total > 0 else 0
        avg_win = daily_returns[daily_returns > 0].mean() if wins > 0 else 0
        avg_loss = abs(daily_returns[daily_returns < 0].mean()) if wins < total else 1e-12
        profit_loss_ratio = avg_win / avg_loss if avg_loss > 0 else 0

        # 交易成本
        total_commission = sum(t.commission for t in self.trade_history)
        total_slippage = sum(t.slippage_cost for t in self.trade_history)

        # 换手率
        daily_turnovers = []
        for trade in self.trade_history:
            daily_turnovers.append(trade.notional / self.initial_capital)
        avg_turnover = np.mean(daily_turnovers) if daily_turnovers else 0

        return BacktestResult(
            initial_capital=self.initial_capital,
            final_capital=nav_curve.iloc[-1],
            total_return=total_return,
            annualized_return=annualized_return,
            annualized_volatility=annualized_vol,
            sharpe_ratio=sharpe,
            max_drawdown=max_dd,
            max_drawdown_duration=max_dd_duration,
            win_rate=win_rate,
            profit_loss_ratio=profit_loss_ratio,
            total_trades=len(self.trade_history),
            turnover_rate=avg_turnover,
            total_commission=total_commission,
            total_slippage=total_slippage,
            daily_returns=daily_returns,
            nav_curve=nav_curve,
            positions_history=self.position_history,
        )
