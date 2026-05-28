"""
策略回测流水线 CLI 与编排模块

使用模型预测得分执行投资组合回测模拟，输出绩效指标。
包装 src.analyzers.portfolio_strategy 中各策略与模拟器。

用法:
    python -m src.workflow.backtest_pipeline --config config/qlib_config.yaml
    python -m src.workflow.backtest_pipeline --strategy topk_dropout --model-path models/lgb_model.pkl
    python -m src.workflow.backtest_pipeline --strategy equal_weight --start 2020-01-01 --end 2023-12-31
"""

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, Optional

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import yaml

from src.analyzers.portfolio_strategy import (
    StrategyConfig,
    TopkDropoutStrategy,
    EqualWeightStrategy,
    ScoreWeightStrategy,
    PortfolioSimulator,
    RiskManager,
)
from src.analyzers.ml_pipeline import BaseForecastModel
from src.utils.logger import get_logger


STRATEGY_MAP = {
    "topk_dropout": TopkDropoutStrategy,
    "equal_weight": EqualWeightStrategy,
    "score_weight": ScoreWeightStrategy,
}


def load_predictions(
    model_path: Optional[str] = None,
    predictions_path: Optional[str] = None,
    data_dir: str = "./data/qlib_data/us_data",
    start_date: str = "2020-01-01",
    end_date: str = "2023-12-31",
) -> pd.DataFrame:
    """
    加载预测得分矩阵。

    优先级: predictions_path > model_path > 模拟数据
    """
    logger = get_logger(__name__)

    # 从文件加载已有预测
    if predictions_path:
        pred_file = Path(predictions_path)
        if pred_file.suffix == ".csv":
            preds = pd.read_csv(predictions_path, index_col=0, parse_dates=True)
        elif pred_file.suffix == ".parquet":
            preds = pd.read_parquet(predictions_path)
        else:
            raise ValueError(f"不支持的预测文件格式: {pred_file.suffix}")
        logger.info("已加载预测", path=predictions_path, shape=preds.shape)
        return preds

    # 从模型生成预测
    if model_path:
        model = BaseForecastModel.load(model_path)
        # 加载测试数据特征
        features_path = Path(data_dir) / "features.parquet"
        if features_path.exists():
            df = pd.read_parquet(features_path)
            feature_cols = [c for c in df.columns if c not in ("datetime", "date", "instrument", "label")]
            X_cols = [c for c in feature_cols if df[c].dtype in ("float64", "float32", "int64", "int32")]
            X = df[X_cols].fillna(0).values.astype(np.float32)
            preds = model.predict(X)
            logger.info("模型预测完成", predictions=len(preds))
            return pd.DataFrame({"score": preds}, index=df.index)
        else:
            logger.warning("特征文件不存在，使用模拟数据")

    # 模拟预测数据
    logger.warning("未指定预测源，使用模拟数据生成回测")
    dates = pd.date_range(start_date, end_date, freq="B")
    instruments = [f"STOCK_{i:03d}" for i in range(100)]
    np.random.seed(42)
    data = np.random.randn(len(dates), len(instruments))
    return pd.DataFrame(data, index=dates, columns=instruments)


def load_prices(
    data_dir: str = "./data/qlib_data/us_data",
    start_date: str = "2020-01-01",
    end_date: str = "2023-12-31",
) -> pd.DataFrame:
    """加载价格数据"""
    logger = get_logger(__name__)

    prices_path = Path(data_dir) / "prices.parquet"
    if prices_path.exists():
        prices = pd.read_parquet(prices_path)
        logger.info("已加载价格数据", path=str(prices_path), shape=prices.shape)
        return prices

    # 模拟价格数据
    logger.warning("价格文件不存在，使用模拟价格数据")
    dates = pd.date_range(start_date, end_date, freq="B")
    instruments = [f"STOCK_{i:03d}" for i in range(100)]
    np.random.seed(123)
    prices = 100 * np.exp(np.random.randn(len(dates), len(instruments)).cumsum(axis=0) * 0.01)
    return pd.DataFrame(prices, index=dates, columns=instruments)


def run_backtest(
    strategy_type: str = "topk_dropout",
    config_path: Optional[str] = None,
    model_path: Optional[str] = None,
    predictions_path: Optional[str] = None,
    data_dir: str = "./data/qlib_data/us_data",
    start_date: str = "2020-01-01",
    end_date: str = "2023-12-31",
    initial_capital: float = 1_000_000,
    output_dir: str = "./experiments",
) -> Dict[str, Any]:
    """
    执行策略回测模拟

    Args:
        strategy_type: 策略类型
        config_path: YAML 配置文件路径
        model_path: 模型路径
        predictions_path: 预测文件路径
        data_dir: 数据目录
        start_date: 回测起始日期
        end_date: 回测截止日期
        initial_capital: 初始资金
        output_dir: 输出目录

    Returns:
        回测结果摘要
    """
    logger = get_logger(__name__)

    # 加载配置
    strategy_config = StrategyConfig()
    if config_path:
        config_file = Path(config_path)
        if config_file.exists():
            with open(config_file, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)
            strategy_cfg = config.get("strategy", {})
            strategy_config = StrategyConfig(
                top_k=strategy_cfg.get("top_k", 30),
                rebalance_freq=strategy_cfg.get("rebalance_freq", 5),
                weight_method=strategy_cfg.get("weight_method", "equal"),
                dropout_threshold=strategy_cfg.get("dropout_threshold", 0.2),
                commission_rate=strategy_cfg.get("commission_rate", 0.001),
                slippage_bps=strategy_cfg.get("slippage_bps", 1.0),
                max_drawdown_limit=strategy_cfg.get("max_drawdown_limit", 0.15),
            )
            logger.info("已加载策略配置", config=strategy_config)

    # 加载预测和价格
    predictions = load_predictions(
        model_path=model_path,
        predictions_path=predictions_path,
        data_dir=data_dir,
        start_date=start_date,
        end_date=end_date,
    )
    prices = load_prices(data_dir, start_date, end_date)

    # 创建策略
    strategy_cls = STRATEGY_MAP.get(strategy_type, TopkDropoutStrategy)
    strategy = strategy_cls(config=strategy_config)
    logger.info("策略已创建", type=strategy_type, top_k=strategy_config.top_k)

    # 创建风险管理和模拟器
    risk_manager = RiskManager(
        max_drawdown_limit=strategy_config.max_drawdown_limit,
    )
    simulator = PortfolioSimulator(
        strategy=strategy,
        initial_capital=initial_capital,
        commission_rate=strategy_config.commission_rate,
        slippage_bps=strategy_config.slippage_bps,
        risk_manager=risk_manager,
    )

    # 执行回测
    result = simulator.run(predictions, prices)

    # 构建结果摘要
    summary = {
        "strategy": strategy_type,
        "initial_capital": initial_capital,
        "total_return": getattr(result, "total_return", 0.0),
        "annual_return": getattr(result, "annual_return", 0.0),
        "sharpe_ratio": getattr(result, "sharpe_ratio", 0.0),
        "max_drawdown": getattr(result, "max_drawdown", 0.0),
        "win_rate": getattr(result, "win_rate", 0.0),
        "total_trades": getattr(result, "total_trades", 0),
        "turnover": getattr(result, "turnover", 0.0),
    }

    logger.info(
        "回测完成",
        total_return=f"{summary['total_return']:.2%}",
        sharpe=f"{summary['sharpe_ratio']:.2f}",
        max_dd=f"{summary['max_drawdown']:.2%}",
    )

    # 保存结果
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    result_file = output_path / f"backtest_{strategy_type}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.json"

    import json
    result_dict = {
        **summary,
        "config": {
            "strategy_type": strategy_type,
            "top_k": strategy_config.top_k,
            "commission_rate": strategy_config.commission_rate,
            "slippage_bps": strategy_config.slippage_bps,
            "start_date": start_date,
            "end_date": end_date,
        },
    }
    with open(result_file, "w", encoding="utf-8") as f:
        json.dump(result_dict, f, indent=2, ensure_ascii=False, default=str)

    logger.info("回测结果已保存", path=str(result_file))

    return summary


def main():
    parser = argparse.ArgumentParser(
        description="策略回测流水线 (Backtest Pipeline)",
    )
    parser.add_argument(
        "--config", "-c",
        default="config/qlib_config.yaml",
        help="配置文件路径",
    )
    parser.add_argument(
        "--strategy", "-s",
        default="topk_dropout",
        choices=["topk_dropout", "equal_weight", "score_weight"],
        help="策略类型",
    )
    parser.add_argument(
        "--model-path", "-m",
        default=None,
        help="模型权重文件路径",
    )
    parser.add_argument(
        "--predictions", "-p",
        default=None,
        help="预测得分文件路径",
    )
    parser.add_argument(
        "--data-dir", "-d",
        default="./data/qlib_data/us_data",
        help="数据目录",
    )
    parser.add_argument(
        "--start-date",
        default="2020-01-01",
        help="回测起始日期",
    )
    parser.add_argument(
        "--end-date",
        default="2023-12-31",
        help="回测截止日期",
    )
    parser.add_argument(
        "--capital",
        type=float,
        default=1_000_000,
        help="初始资金 (默认: 1,000,000)",
    )
    parser.add_argument(
        "--output-dir", "-o",
        default="./experiments",
        help="结果输出目录",
    )

    args = parser.parse_args()

    logger = get_logger(__name__)

    try:
        summary = run_backtest(
            strategy_type=args.strategy,
            config_path=args.config,
            model_path=args.model_path,
            predictions_path=args.predictions,
            data_dir=args.data_dir,
            start_date=args.start_date,
            end_date=args.end_date,
            initial_capital=args.capital,
            output_dir=args.output_dir,
        )

        print(f"\nBacktest Summary [{args.strategy}]:")
        print(f"  Total Return:  {summary['total_return']:.2%}")
        print(f"  Annual Return: {summary['annual_return']:.2%}")
        print(f"  Sharpe Ratio:  {summary['sharpe_ratio']:.2f}")
        print(f"  Max Drawdown:  {summary['max_drawdown']:.2%}")
        print(f"  Win Rate:      {summary['win_rate']:.2%}")
        print(f"  Total Trades:  {summary['total_trades']}")

    except Exception as e:
        logger.error(f"回测流水线执行失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
