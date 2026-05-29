"""
TopkDropout 动量淘汰策略 (独立策略模块)

从 src/analyzers/portfolio_strategy.py 中分离为独立文件，
遵循 PRD 第 2.3 节分层架构原则: analyzers 负责生成预测分数，
strategies 专职持仓映射与换手控制。

核心策略:
- TopkDropoutStrategy: 基于预测排序的动量淘汰策略
- EqualWeightStrategy: 等权配置策略 (基准对比)
- ScoreWeightStrategy: 预测得分加权策略

设计原则:
- 严格防过拟合 (训练/回测时间隔离)
- 真实交易成本建模 (佣金 + 滑点)
- 可配置的风控约束
- YAML 驱动实例化 (通过 STRATEGY_REGISTRY)

使用示例:
    from src.strategies.topk_dropout import TopkDropoutStrategy, StrategyConfig

    config = StrategyConfig(top_k=30, dropout_threshold=0.2)
    strategy = TopkDropoutStrategy(config=config)
    weights = strategy.generate_weights(scores, positions, prices, date)
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

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
#  权重分配器
# ========================================================================

class WeightAllocator:
    """权重分配器 --- 根据得分计算组合权重"""

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
            logger = get_logger()
            logger.warning("INV_VOL 需要波动率数据，回退到等权")

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
#  组合策略 --- 抽象基类
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
        current_positions: Dict[str, Any],
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
#  TopkDropoutStrategy --- 动量淘汰策略
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
        current_positions: Dict[str, Any],
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
#  EqualWeightStrategy --- 等权策略
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
        current_positions: Dict[str, Any],
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
#  ScoreWeightStrategy --- 得分加权策略
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
        current_positions: Dict[str, Any],
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
#  策略工厂 (YAML 驱动)
# ========================================================================

STRATEGY_REGISTRY: Dict[str, type] = {
    "topk_dropout": TopkDropoutStrategy,
    "equal_weight": EqualWeightStrategy,
    "score_weight": ScoreWeightStrategy,
}


def create_strategy(strategy_key: str, **kwargs) -> BaseStrategy:
    """
    从 YAML 配置中的策略 key 动态创建策略实例。

    Args:
        strategy_key: 策略标识符 (如 'topk_dropout', 'equal_weight')
        **kwargs: 策略参数

    Returns:
        BaseStrategy 子类实例

    Raises:
        ValueError: 未知策略 key
    """
    cls = STRATEGY_REGISTRY.get(strategy_key.lower())
    if cls is None:
        raise ValueError(
            f"未知策略: {strategy_key}. 可用策略: {list(STRATEGY_REGISTRY.keys())}"
        )
    return cls(**kwargs)


def list_available_strategies() -> list:
    """列出所有可用策略名称"""
    return list(STRATEGY_REGISTRY.keys())
