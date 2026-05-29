"""
组合策略模块 (Portfolio Strategy Module)

按 PRD 第 2.3 节架构规范，策略决策与因子分析解耦。
此模块专职持仓映射、换手控制和交易执行逻辑。

包含:
- TopkDropoutStrategy: 基于预测排序的动量淘汰策略
- EqualWeightStrategy: 等权配置策略 (基准对比)
- ScoreWeightStrategy: 预测得分加权策略
- WeightAllocator: 权重分配器 (等权/得分/排名/逆波动率)
- RiskManager: 风控模块 (最大回撤熔断/止损)

策略工厂:
    通过 STRATEGY_REGISTRY 支持从 YAML 配置动态实例化，
    与 runner.py 工作流通过配置驱动接口调用。

使用示例:
    from src.strategies import TopkDropoutStrategy, StrategyConfig

    config = StrategyConfig(top_k=30, dropout_threshold=0.2)
    strategy = TopkDropoutStrategy(config=config)
"""

from src.strategies.topk_dropout import (
    BaseStrategy,
    EqualWeightStrategy,
    RiskManager,
    ScoreWeightStrategy,
    SignalType,
    StrategyConfig,
    TopkDropoutStrategy,
    WeightAllocator,
    WeightMethod,
    STRATEGY_REGISTRY,
    create_strategy,
    list_available_strategies,
)

__all__ = [
    "BaseStrategy",
    "TopkDropoutStrategy",
    "EqualWeightStrategy",
    "ScoreWeightStrategy",
    "WeightAllocator",
    "RiskManager",
    "StrategyConfig",
    "WeightMethod",
    "SignalType",
    "STRATEGY_REGISTRY",
    "create_strategy",
    "list_available_strategies",
]
