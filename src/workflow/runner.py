"""
工作流编排引擎 (Workflow Orchestrator)

Qlib 风格的 YAML 配置驱动实验流水线，扮演系统业务逻辑的
全局总导演角色。串联数据加载 → 特征处理 → 模型训练 →
回测评估 → 报告生成的全生命周期。

核心组件:
- ExperimentConfig: YAML 配置解析器
- WorkflowOrchestrator: 实验流水线编排器
- ExperimentTracker: 实验记录与复现
- CLIRunner: 命令行入口 (qrun 风格)

配置文件示例:
    experiment:
      name: "lightgbm_baseline"
      description: "LightGBM 基线实验"
    
    data:
      train_start: "2015-01-01"
      train_end: "2020-12-31"
      valid_start: "2021-01-01"
      valid_end: "2021-12-31"
      test_start: "2022-01-01"
      test_end: "2023-12-31"
    
    model:
      type: "lightgbm"
      params:
        num_leaves: 64
        learning_rate: 0.05
        n_estimators: 1000
    
    strategy:
      type: "topk_dropout"
      top_k: 30
      rebalance_freq: 5
    
    processors:
      - type: "winsorize"
        limits: [0.01, 0.99]
      - type: "fillna"
        strategy: "cross_sectional"
      - type: "robust_zscore"

使用方式:
    python -m src.workflow.runner --config experiments/lgb_baseline.yaml
"""

import json
import os
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import yaml

from src.utils.logger import get_logger


# ========================================================================
#  配置数据类
# ========================================================================

@dataclass
class DataConfig:
    """数据配置"""
    train_start: str = "2015-01-01"
    train_end: str = "2019-12-31"
    valid_start: str = "2020-01-01"
    valid_end: str = "2020-12-31"
    test_start: str = "2021-01-01"
    test_end: str = "2023-12-31"
    instruments: List[str] = field(default_factory=list)
    fields: List[str] = field(default_factory=list)
    label_col: str = "label"
    label_period: int = 20  # 未来N日收益率


@dataclass
class ModelConfig:
    """模型配置"""
    type: str = "lightgbm"
    params: Dict[str, Any] = field(default_factory=dict)
    checkpoint_dir: str = "models/checkpoints"


@dataclass
class StrategyConfig:
    """策略配置"""
    type: str = "topk_dropout"
    top_k: int = 30
    rebalance_freq: int = 5
    weight_method: str = "equal"
    dropout_threshold: float = 0.2
    commission_rate: float = 0.001
    slippage_bps: float = 1.0
    max_drawdown_limit: float = 0.15


@dataclass
class ExperimentConfig:
    """实验配置"""
    name: str = "default_experiment"
    description: str = ""
    version: str = "1.0"
    seed: int = 42
    output_dir: str = "experiments"
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    strategy: StrategyConfig = field(default_factory=StrategyConfig)
    processors: List[Dict[str, Any]] = field(default_factory=list)
    factors: List[str] = field(default_factory=list)

    @classmethod
    def from_yaml(cls, path: str) -> "ExperimentConfig":
        """从 YAML 文件加载配置"""
        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)

        exp = raw.get("experiment", {})
        data_raw = raw.get("data", {})
        model_raw = raw.get("model", {})
        strategy_raw = raw.get("strategy", {})

        return cls(
            name=exp.get("name", "default_experiment"),
            description=exp.get("description", ""),
            version=exp.get("version", "1.0"),
            seed=exp.get("seed", 42),
            output_dir=exp.get("output_dir", "experiments"),
            data=DataConfig(
                train_start=data_raw.get("train_start", "2015-01-01"),
                train_end=data_raw.get("train_end", "2019-12-31"),
                valid_start=data_raw.get("valid_start", "2020-01-01"),
                valid_end=data_raw.get("valid_end", "2020-12-31"),
                test_start=data_raw.get("test_start", "2021-01-01"),
                test_end=data_raw.get("test_end", "2023-12-31"),
                instruments=data_raw.get("instruments", []),
                fields=data_raw.get("fields", []),
                label_col=data_raw.get("label_col", "label"),
                label_period=data_raw.get("label_period", 20),
            ),
            model=ModelConfig(
                type=model_raw.get("type", "lightgbm"),
                params=model_raw.get("params", {}),
                checkpoint_dir=model_raw.get("checkpoint_dir", "models/checkpoints"),
            ),
            strategy=StrategyConfig(
                type=strategy_raw.get("type", "topk_dropout"),
                top_k=strategy_raw.get("top_k", 30),
                rebalance_freq=strategy_raw.get("rebalance_freq", 5),
                weight_method=strategy_raw.get("weight_method", "equal"),
                dropout_threshold=strategy_raw.get("dropout_threshold", 0.2),
                commission_rate=strategy_raw.get("commission_rate", 0.001),
                slippage_bps=strategy_raw.get("slippage_bps", 1.0),
                max_drawdown_limit=strategy_raw.get("max_drawdown_limit", 0.15),
            ),
            processors=raw.get("processors", []),
            factors=raw.get("factors", []),
        )

    def to_yaml(self, path: str):
        """保存配置为 YAML"""
        data = {
            "experiment": {
                "name": self.name,
                "description": self.description,
                "version": self.version,
                "seed": self.seed,
                "output_dir": self.output_dir,
            },
            "data": asdict(self.data),
            "model": {
                "type": self.model.type,
                "params": self.model.params,
                "checkpoint_dir": self.model.checkpoint_dir,
            },
            "strategy": asdict(self.strategy),
            "processors": self.processors,
            "factors": self.factors,
        }
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(data, f, default_flow_style=False, allow_unicode=True)


# ========================================================================
#  实验追踪器
# ========================================================================

@dataclass
class ExperimentRecord:
    """实验运行记录"""
    experiment_id: str
    config: ExperimentConfig
    started_at: str = ""
    finished_at: str = ""
    status: str = "pending"  # pending | running | completed | failed
    metrics: Dict[str, Any] = field(default_factory=dict)
    artifacts: List[str] = field(default_factory=list)
    error: Optional[str] = None
    duration_seconds: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "config": asdict(self.config),
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "status": self.status,
            "metrics": self.metrics,
            "artifacts": self.artifacts,
            "error": self.error,
            "duration_seconds": self.duration_seconds,
        }


class ExperimentTracker:
    """
    实验追踪器 — 记录每次实验的配置、指标和产物

    支持:
    - 实验记录持久化 (JSON)
    - 实验复现 (基于配置重建)
    - 多实验对比
    """

    def __init__(self, log_dir: str = "experiments/logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.logger = get_logger()

    def create_experiment(self, config: ExperimentConfig) -> ExperimentRecord:
        """创建新实验记录"""
        exp_id = f"{config.name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
        record = ExperimentRecord(
            experiment_id=exp_id,
            config=config,
            started_at=datetime.now().isoformat(),
            status="running",
        )
        self._save_record(record)
        self.logger.info("实验已创建", experiment_id=exp_id, name=config.name)
        return record

    def complete_experiment(self, record: ExperimentRecord, metrics: Dict[str, Any]):
        """标记实验完成并记录指标"""
        record.status = "completed"
        record.finished_at = datetime.now().isoformat()
        record.metrics = metrics
        if record.started_at:
            start = datetime.fromisoformat(record.started_at)
            record.duration_seconds = (datetime.now() - start).total_seconds()
        self._save_record(record)
        self.logger.info("实验已完成", experiment_id=record.experiment_id,
                         duration_s=round(record.duration_seconds, 1))

    def fail_experiment(self, record: ExperimentRecord, error: str):
        """标记实验失败"""
        record.status = "failed"
        record.finished_at = datetime.now().isoformat()
        record.error = error
        self._save_record(record)
        self.logger.error("实验失败", experiment_id=record.experiment_id, error=error)

    def get_experiment(self, experiment_id: str) -> Optional[ExperimentRecord]:
        """获取实验记录"""
        path = self.log_dir / f"{experiment_id}.json"
        if not path.exists():
            return None
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return self._dict_to_record(data)

    def list_experiments(self, limit: int = 50) -> List[Dict[str, Any]]:
        """列出最近实验"""
        files = sorted(self.log_dir.glob("*.json"), key=os.path.getmtime, reverse=True)
        results = []
        for f in files[:limit]:
            with open(f, "r", encoding="utf-8") as fp:
                data = json.load(fp)
            results.append({
                "experiment_id": data.get("experiment_id"),
                "status": data.get("status"),
                "started_at": data.get("started_at"),
                "duration_seconds": data.get("duration_seconds", 0),
                "metrics": data.get("metrics", {}),
            })
        return results

    def compare_experiments(self, experiment_ids: List[str]) -> pd.DataFrame:
        """对比多个实验的指标"""
        rows = []
        for eid in experiment_ids:
            record = self.get_experiment(eid)
            if record:
                row = {
                    "experiment_id": eid,
                    "status": record.status,
                    **record.metrics,
                }
                rows.append(row)
        return pd.DataFrame(rows)

    def _save_record(self, record: ExperimentRecord):
        path = self.log_dir / f"{record.experiment_id}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(record.to_dict(), f, indent=2, ensure_ascii=False)

    def _dict_to_record(self, data: Dict[str, Any]) -> ExperimentRecord:
        config_raw = data.get("config", {})
        config = ExperimentConfig(
            name=config_raw.get("name", ""),
            seed=config_raw.get("seed", 42),
            data=DataConfig(**config_raw.get("data", {})),
            model=ModelConfig(**config_raw.get("model", {})),
            strategy=StrategyConfig(**config_raw.get("strategy", {})),
            processors=config_raw.get("processors", []),
            factors=config_raw.get("factors", []),
        )
        return ExperimentRecord(
            experiment_id=data.get("experiment_id", ""),
            config=config,
            started_at=data.get("started_at", ""),
            finished_at=data.get("finished_at", ""),
            status=data.get("status", "pending"),
            metrics=data.get("metrics", {}),
            artifacts=data.get("artifacts", []),
            error=data.get("error"),
            duration_seconds=data.get("duration_seconds", 0),
        )


# ========================================================================
#  工作流编排器
# ========================================================================

class WorkflowOrchestrator:
    """
    工作流编排器 — 串联完整实验流水线

    流水线阶段:
    1. 数据准备 (Data Preparation)
    2. 特征工程 (Feature Engineering)
    3. 模型训练 (Model Training)
    4. 预测生成 (Prediction)
    5. 组合回测 (Backtest)
    6. 报告生成 (Reporting)

    使用示例:
        orchestrator = WorkflowOrchestrator()
        config = ExperimentConfig.from_yaml("experiments/lgb_baseline.yaml")
        result = orchestrator.run(config)
    """

    def __init__(self):
        self.logger = get_logger()
        self.tracker = ExperimentTracker()

    def run(self, config: ExperimentConfig) -> ExperimentRecord:
        """
        执行完整实验流水线

        Args:
            config: 实验配置

        Returns:
            ExperimentRecord
        """
        record = self.tracker.create_experiment(config)
        exp_dir = Path(config.output_dir) / record.experiment_id
        exp_dir.mkdir(parents=True, exist_ok=True)

        # 保存配置快照
        config.to_yaml(str(exp_dir / "config.yaml"))

        try:
            # ===== 阶段1: 数据准备 =====
            self.logger.info("=== 阶段1: 数据准备 ===", experiment=record.experiment_id)
            train_data, valid_data, test_data = self._prepare_data(config)

            # ===== 阶段2: 特征工程 =====
            self.logger.info("=== 阶段2: 特征工程 ===", experiment=record.experiment_id)
            X_train, y_train, X_valid, y_valid, X_test, y_test = self._process_features(
                config, train_data, valid_data, test_data
            )

            # ===== 阶段3: 模型训练 =====
            self.logger.info("=== 阶段3: 模型训练 ===", experiment=record.experiment_id)
            model, training_result = self._train_model(config, X_train, y_train, X_valid, y_valid)

            # ===== 阶段4: 预测生成 =====
            self.logger.info("=== 阶段4: 预测生成 ===", experiment=record.experiment_id)
            predictions = self._generate_predictions(config, model, X_test, test_data)

            # ===== 阶段5: 组合回测 =====
            self.logger.info("=== 阶段5: 组合回测 ===", experiment=record.experiment_id)
            backtest_result = self._run_backtest(config, predictions, test_data)

            # ===== 阶段6: 报告生成 =====
            self.logger.info("=== 阶段6: 报告生成 ===", experiment=record.experiment_id)
            report_metrics = self._generate_report(
                config, training_result, backtest_result, exp_dir
            )

            # 汇总指标
            metrics = {
                "ic_mean": getattr(training_result, "best_score", 0),
                "total_return": getattr(backtest_result, "total_return", 0) if backtest_result else 0,
                "sharpe_ratio": getattr(backtest_result, "sharpe_ratio", 0) if backtest_result else 0,
                "max_drawdown": getattr(backtest_result, "max_drawdown", 0) if backtest_result else 0,
                "n_features": getattr(training_result, "n_features", 0),
                **report_metrics,
            }

            # 保存模型
            model_path = str(exp_dir / "model.pkl")
            if hasattr(model, "save"):
                model.save(model_path)
                record.artifacts.append(model_path)

            self.tracker.complete_experiment(record, metrics)

        except Exception as e:
            self.logger.error("实验执行失败", experiment=record.experiment_id, error=str(e))
            self.tracker.fail_experiment(record, str(e))
            raise

        return record

    def _prepare_data(
        self, config: ExperimentConfig
    ) -> Tuple[Any, Any, Any]:
        """
        数据准备阶段

        从数据源加载特征矩阵并按时间切分训练/验证/测试集。
        实际实现应查询 .bin 文件或 PIT 数据库。
        """
        # 阶段输出: (train_df, valid_df, test_df)
        # 当前为占位实现
        self.logger.info("数据准备 (占位)", train_period=f"{config.data.train_start}~{config.data.train_end}")
        return None, None, None

    def _process_features(
        self,
        config: ExperimentConfig,
        train_data: Any,
        valid_data: Any,
        test_data: Any,
    ) -> Tuple[Any, Any, Any, Any, Any, Any]:
        """
        特征工程阶段

        根据配置的处理器链对特征矩阵进行清洗和标准化。
        """
        if config.processors:
            self.logger.info("特征处理链", processors=[p.get("type") for p in config.processors])

        # 阶段输出: (X_train, y_train, X_valid, y_valid, X_test, y_test)
        return None, None, None, None, None, None

    def _train_model(
        self,
        config: ExperimentConfig,
        X_train: Any,
        y_train: Any,
        X_valid: Any,
        y_valid: Any,
    ) -> Tuple[Any, Any]:
        """
        模型训练阶段

        根据配置创建模型实例并执行训练。
        """
        from src.analyzers.ml_pipeline import BaseForecastModel, MLPipeline

        self.logger.info("模型训练", type=config.model.type, params=config.model.params)

        model = BaseForecastModel.create(config.model.type, **config.model.params)
        pipeline = MLPipeline(model)

        # 阶段输出: (model, training_result)
        return model, None

    def _generate_predictions(
        self,
        config: ExperimentConfig,
        model: Any,
        X_test: Any,
        test_data: Any,
    ) -> pd.DataFrame:
        """
        预测生成阶段

        在测试集上生成横截面预测得分。
        """
        self.logger.info("预测生成", model=config.model.type)
        # 阶段输出: predictions DataFrame
        return pd.DataFrame()

    def _run_backtest(
        self,
        config: ExperimentConfig,
        predictions: pd.DataFrame,
        test_data: Any,
    ) -> Any:
        """
        组合回测阶段

        根据配置的策略类型执行回测模拟。
        """
        from src.analyzers.portfolio_strategy import (
            StrategyConfig,
            TopkDropoutStrategy,
            EqualWeightStrategy,
            ScoreWeightStrategy,
            PortfolioSimulator,
        )

        strategy_map = {
            "topk_dropout": TopkDropoutStrategy,
            "equal_weight": EqualWeightStrategy,
            "score_weight": ScoreWeightStrategy,
        }

        strategy_cls = strategy_map.get(config.strategy.type, TopkDropoutStrategy)
        strategy_config = StrategyConfig(
            top_k=config.strategy.top_k,
            rebalance_freq=config.strategy.rebalance_freq,
            weight_method=config.strategy.weight_method,
            dropout_threshold=config.strategy.dropout_threshold,
            commission_rate=config.strategy.commission_rate,
            slippage_bps=config.strategy.slippage_bps,
            max_drawdown_limit=config.strategy.max_drawdown_limit,
        )
        strategy = strategy_cls(config=strategy_config)

        self.logger.info("组合回测", strategy=config.strategy.type, top_k=config.strategy.top_k)

        # 阶段输出: BacktestResult
        return None

    def _generate_report(
        self,
        config: ExperimentConfig,
        training_result: Any,
        backtest_result: Any,
        exp_dir: Path,
    ) -> Dict[str, Any]:
        """
        报告生成阶段

        生成 HTML/JSON 绩效报告和图表。
        """
        self.logger.info("报告生成", output_dir=str(exp_dir))

        report_metrics: Dict[str, Any] = {}

        # 导出配置为 YAML (供复现)
        config_path = exp_dir / "config.yaml"
        config.to_yaml(str(config_path))

        return report_metrics


# ========================================================================
#  命令行入口 (qrun 风格)
# ========================================================================

def run_experiment_from_yaml(config_path: str) -> ExperimentRecord:
    """
    从 YAML 配置文件运行实验

    类似 Qlib 的 qrun 命令:
        python -m src.workflow.runner --config experiments/lgb_baseline.yaml

    Args:
        config_path: YAML 配置文件路径

    Returns:
        ExperimentRecord
    """
    logger = get_logger()
    logger.info("加载实验配置", path=config_path)

    config = ExperimentConfig.from_yaml(config_path)
    orchestrator = WorkflowOrchestrator()

    record = orchestrator.run(config)

    logger.info("实验执行完成",
                experiment=record.experiment_id,
                status=record.status,
                duration=round(record.duration_seconds, 1))

    return record
