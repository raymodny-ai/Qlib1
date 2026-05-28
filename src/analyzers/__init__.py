"""
智能分析引擎与因子计算网络模块

包含:
- Expression Engine: AST 表达式编译器与求值器
- Alpha Factors: 基本面/技术面因子库
- ML Pipeline: 机器学习模型训练管道
- Portfolio Strategy: 组合优化与回测策略
- Accuracy Validator: 准确度校验与策略红线验证
"""

from src.analyzers.expression_engine import (
    ExpressionCompiler,
    ExpressionEngine,
    CompiledExpression,
)

from src.analyzers.alpha_factors import AlphaFactorCalculator

from src.analyzers.ml_pipeline import (
    BaseForecastModel,
    LightGBMModel,
    XGBoostModel,
    MLPipeline,
    TimeSeriesSplitter,
    TrainingResult,
    PredictionResult,
)

from src.models.adarnn_model import AdaRNNModel
from src.models.tabnet_model import TabNetModel
from src.models.double_ensemble_model import DoubleEnsembleModel

from src.analyzers.portfolio_strategy import (
    BaseStrategy,
    TopkDropoutStrategy,
    EqualWeightStrategy,
    ScoreWeightStrategy,
    PortfolioSimulator,
    RiskManager,
    WeightAllocator,
    StrategyConfig,
    WeightMethod,
    SignalType,
    BacktestResult,
    TradeRecord,
    Position,
)

from src.analyzers.report_generator import (
    BacktestAnalyzer,
    PerformanceReport,
    ReportExporter,
    FullReport,
    ICMetrics,
    ReturnMetrics,
    RiskMetrics,
)

from src.analyzers.accuracy_validator import (
    AccuracyThresholdValidator,
    RollingICValidator,
    DrawdownValidator,
    StrategyStabilityChecker,
    ValidationReport,
    ThresholdCheck,
    CheckSeverity,
    ThresholdPreset,
    quick_validate,
)

__all__ = [
    # Expression Engine
    "ExpressionCompiler",
    "ExpressionEngine",
    "CompiledExpression",
    # Alpha Factors
    "AlphaFactorCalculator",
    # ML Pipeline
    "BaseForecastModel",
    "LightGBMModel",
    "XGBoostModel",
    "MLPipeline",
    "TimeSeriesSplitter",
    "TrainingResult",
    "PredictionResult",
    # Deep Models
    "AdaRNNModel",
    "TabNetModel",
    "DoubleEnsembleModel",
    # Portfolio Strategy
    "BaseStrategy",
    "TopkDropoutStrategy",
    "EqualWeightStrategy",
    "ScoreWeightStrategy",
    "PortfolioSimulator",
    "RiskManager",
    "WeightAllocator",
    "StrategyConfig",
    "WeightMethod",
    "SignalType",
    "BacktestResult",
    "TradeRecord",
    "Position",
    # Report Generator
    "BacktestAnalyzer",
    "PerformanceReport",
    "ReportExporter",
    "FullReport",
    "ICMetrics",
    "ReturnMetrics",
    "RiskMetrics",
    # Accuracy Validator
    "AccuracyThresholdValidator",
    "RollingICValidator",
    "DrawdownValidator",
    "StrategyStabilityChecker",
    "ValidationReport",
    "ThresholdCheck",
    "CheckSeverity",
    "ThresholdPreset",
    "quick_validate",
]
