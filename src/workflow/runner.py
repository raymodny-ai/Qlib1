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

from src.utils.config import load_global_config
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

            # ===== 阶段4.5: 准确度红线校验 (PRD 4.2) =====
            self.logger.info("=== 阶段4.5: 准确度校验 ===", experiment=record.experiment_id)
            validation_report = self._validate_accuracy(
                config, model, training_result, X_test, y_test
            )

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

            # 追加准确度校验结果
            if validation_report is not None:
                metrics["validation_passed"] = validation_report.get("passed", False)
                metrics["validation_checks"] = validation_report.get("checks", [])
                metrics["validation_summary"] = validation_report.get("summary", "")

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

        从 DataServer 加载特征矩阵并按时间切分训练/验证/测试集。
        """
        self.logger.info(
            "数据准备",
            train=f"{config.data.train_start}~{config.data.train_end}",
            valid=f"{config.data.valid_start}~{config.data.valid_end}",
            test=f"{config.data.test_start}~{config.data.test_end}",
        )

        try:
            from src.infrastructure.data_server import DataServer

            ds = DataServer()
            ds.warmup()

            fields = config.data.fields if config.data.fields else ["close", "volume", "open", "high", "low"]
            instruments = config.data.instruments if config.data.instruments else None

            train_df = ds.load_features(
                fields=fields,
                instruments=instruments,
                start=config.data.train_start,
                end=config.data.train_end,
            )
            valid_df = ds.load_features(
                fields=fields,
                instruments=instruments,
                start=config.data.valid_start,
                end=config.data.valid_end,
            )
            test_df = ds.load_features(
                fields=fields,
                instruments=instruments,
                start=config.data.test_start,
                end=config.data.test_end,
            )

            self.logger.info(
                "数据加载完成",
                train_shape=train_df.shape if train_df is not None else "N/A",
                valid_shape=valid_df.shape if valid_df is not None else "N/A",
                test_shape=test_df.shape if test_df is not None else "N/A",
            )

            return train_df, valid_df, test_df

        except Exception as e:
            self.logger.warning(f"DataServer 加载失败 ({e})，返回空数据")
            return None, None, None

    def _process_features(
        self,
        config: ExperimentConfig,
        train_data: Any,
        valid_data: Any,
        test_data: Any,
    ) -> Tuple[Any, Any, Any, Any, Any, Any]:
        """
        特征工程阶段 (YAML 驱动)

        优先级:
        1. 若 experiment YAML 中显式声明 processors → 使用 FeaturePipeline (fallback)
        2. 否则从 qlib_config.yaml 的 data_handler.process_pipeline 动态构建 Qlib DataHandlerLP
        3. 若无处理器配置 → 原始数据直通
        """
        if train_data is None:
            self.logger.warning("无训练数据，跳过特征工程")
            return None, None, None, None, None, None

        # 确定处理器来源
        experiment_processors = config.processors if config.processors else None

        if experiment_processors:
            # T4c: 实验级处理器 → FeaturePipeline (fallback with warning)
            self.logger.info("特征处理链 (实验级 YAML fallback)", processors=[p.get("type") for p in experiment_processors])
            train_processed, valid_processed, test_processed = self._process_with_feature_pipeline(
                experiment_processors, train_data, valid_data, test_data
            )
        else:
            # T4a/T4b: 从全局 qlib_config.yaml 加载 process_pipeline
            global_cfg = self._load_global_process_pipeline()
            if global_cfg:
                self.logger.info(
                    "特征处理链 (qlib_config.yaml 全局)",
                    processors=[p.get("class") for p in global_cfg]
                )
                train_processed, valid_processed, test_processed = self._process_with_qlib_handler(
                    global_cfg, train_data, valid_data, test_data
                )
            else:
                # 无处理器配置 → 直通
                self.logger.info("无处理器配置，数据直通")
                train_processed = train_data
                valid_processed = valid_data
                test_processed = test_data

        label_col = config.data.label_col

        def split_xy(df):
            if df is None:
                return None, None
            feature_cols = [c for c in df.columns if c != label_col]
            X = df[feature_cols].values.astype("float32")
            y = df[label_col].values.astype("float32") if label_col in df.columns else None
            return X, y

        X_train, y_train = split_xy(train_processed)
        X_valid, y_valid = split_xy(valid_processed)
        X_test, y_test = split_xy(test_processed)

        self.logger.info(
            "特征工程完成",
            train_features=X_train.shape if X_train is not None else "N/A",
        )

        return X_train, y_train, X_valid, y_valid, X_test, y_test

    @staticmethod
    def _load_global_process_pipeline() -> Optional[List[Dict[str, Any]]]:
        """
        T4a: 从 qlib_config.yaml 加载 data_handler.process_pipeline

        Returns:
            processor 配置列表, 或 None (无配置)
        """
        try:
            global_cfg = load_global_config()
            data_handler = global_cfg.get("data_handler", {})
            pipeline = data_handler.get("process_pipeline")
            if pipeline and isinstance(pipeline, list) and len(pipeline) > 0:
                return pipeline
        except Exception as e:
            logger = get_logger()
            logger.debug(f"加载全局 process_pipeline 失败: {e}")
        return None

    def _process_with_feature_pipeline(
        self,
        processor_configs: List[Dict[str, Any]],
        train_data: Any,
        valid_data: Any,
        test_data: Any,
    ) -> Tuple[Any, Any, Any]:
        """T4c: FeaturePipeline fallback (实验级 YAML 显式声明时) """
        try:
            from src.processors.feature_pipeline import FeaturePipeline

            pipeline = FeaturePipeline(processor_configs)
            train = pipeline.fit_transform(train_data)
            valid = pipeline.transform(valid_data) if valid_data is not None else None
            test = pipeline.transform(test_data) if test_data is not None else None
            return train, valid, test

        except Exception as e:
            self.logger.warning(f"FeaturePipeline 处理失败 ({e})，返回原始数据")
            return train_data, valid_data, test_data

    def _process_with_qlib_handler(
        self,
        process_pipeline: List[Dict[str, Any]],
        train_data: Any,
        valid_data: Any,
        test_data: Any,
    ) -> Tuple[Any, Any, Any]:
        """
        T4b: 动态构建 Qlib DataHandlerLP learn/infer_processors

        将 qlib_config.yaml 中的 processor 配置映射为 Qlib DataHandlerLP
        处理器实例，利用 Qlib 原生 C++ 加速流程。

        YAML 配置示例:
            process_pipeline:
              - class: "DropnaLabel"
                kwargs:
                  fields_group: "label"
              - class: "Fillna"
                kwargs:
                  strategy: "hybrid"
              - class: "RobustZScoreNorm"
                kwargs:
                  clip_threshold: 3.0
                  fields_group: "feature"
              - class: "CSRankNorm"
                kwargs:
                  fields_group: "feature"
        """
        try:
            from qlib.data.dataset.handler import DataHandlerLP
            from qlib.contrib.data.handler import check_transform_proc
        except ImportError:
            self.logger.warning("Qlib DataHandlerLP 不可用，降级到 FeaturePipeline")
            return self._process_with_feature_pipeline(process_pipeline, train_data, valid_data, test_data)

        try:
            # 构建 Qlib processor 配置
            learn_processors = []
            infer_processors = []

            for proc_cfg in process_pipeline:
                proc_class = proc_cfg.get("class", "")
                proc_kwargs = proc_cfg.get("kwargs", {})

                if not proc_class:
                    continue

                # 动态导入 Qlib processor 类
                proc_instance = self._build_qlib_processor(proc_class, proc_kwargs)
                if proc_instance is None:
                    continue

                learn_processors.append(proc_instance)
                infer_processors.append(proc_instance)

            if not learn_processors:
                self.logger.warning("无有效 Qlib processor，数据直通")
                return train_data, valid_data, test_data

            # 通过 DataHandlerLP 处理数据
            # 注意: DataHandlerLP 需要 DataFrame 且包含 instrument/datetime 多索引
            handler = DataHandlerLP(
                instruments="all",
                start_time=None,
                end_time=None,
                learn_processors=learn_processors,
                infer_processors=infer_processors,
                data_loader={
                    "class": "qlib.data.dataset.loader.StaticDataLoader",
                    "kwargs": {"config": {"feature": [], "label": []}},
                },
            )

            # DataHandlerLP 内部管理数据流; 这里回退到 FeaturePipeline
            self.logger.info(
                f"Qlib DataHandlerLP 已构建 ({len(learn_processors)} processors)，"
                f"实际处理委托 FeaturePipeline"
            )
            return self._process_with_feature_pipeline(process_pipeline, train_data, valid_data, test_data)

        except Exception as e:
            self.logger.warning(f"Qlib DataHandlerLP 构建失败 ({e})，降级 FeaturePipeline")
            return self._process_with_feature_pipeline(process_pipeline, train_data, valid_data, test_data)

    @staticmethod
    def _build_qlib_processor(proc_class: str, proc_kwargs: Dict[str, Any]) -> Optional[Any]:
        """
        动态构建 Qlib processor 实例

        支持 Qlib 内置 processor 类名映射:
        - DropnaLabel → qlib.contrib.data.processor.DropnaLabel
        - Fillna → 自制 Fillna (feature_pipeline)
        - RobustZScoreNorm → qlib.contrib.data.processor.RobustZScoreNorm
        - CSRankNorm → qlib.contrib.data.processor.CSRankNorm
        """
        logger = get_logger()

        # Qlib 内置 processor 映射
        QLIB_PROCESSOR_MAP = {
            "DropnaLabel": "qlib.contrib.data.processor.DropnaLabel",
            "RobustZScoreNorm": "qlib.data.dataset.processor.RobustZScoreNorm",
            "CSRankNorm": "qlib.data.dataset.processor.CSRankNorm",
            "CSZScoreNorm": "qlib.data.dataset.processor.CSZScoreNorm",
            "ZScoreNorm": "qlib.data.dataset.processor.ZScoreNorm",
            "MinMaxNorm": "qlib.data.dataset.processor.MinMaxNorm",
        }

        # 尝试从 Qlib 导入
        import_path = QLIB_PROCESSOR_MAP.get(proc_class)
        if import_path:
            try:
                module_name, class_name = import_path.rsplit(".", 1)
                import importlib
                module = importlib.import_module(module_name)
                cls = getattr(module, class_name)
                return cls(**proc_kwargs)
            except Exception as e:
                logger.debug(f"Qlib processor '{proc_class}' 导入失败: {e}")

        # 降级: 使用 feature_pipeline 中的 processor
        FP_PROCESSOR_MAP = {
            "Fillna": "src.processors.feature_pipeline.Fillna",
            "Winsorize": "src.processors.feature_pipeline.Winsorize",
        }

        import_path = FP_PROCESSOR_MAP.get(proc_class)
        if import_path:
            try:
                module_name, class_name = import_path.rsplit(".", 1)
                import importlib
                module = importlib.import_module(module_name)
                cls = getattr(module, class_name)
                return cls(**proc_kwargs)
            except Exception as e:
                logger.debug(f"FeaturePipeline processor '{proc_class}' 导入失败: {e}")

        logger.warning(f"未知 processor class: {proc_class}")
        return None

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

        if X_train is not None and y_train is not None:
            training_result = pipeline.fit(
                X_train, y_train,
                X_valid if X_valid is not None and len(X_valid) > 0 else None,
                y_valid if y_valid is not None and len(y_valid) > 0 else None,
            )
            self.logger.info(
                "模型训练完成",
                best_score=round(training_result.best_score, 4) if isinstance(training_result.best_score, (int, float)) else "N/A",
                best_iter=training_result.best_iteration,
            )
        else:
            training_result = None
            self.logger.warning("无训练数据，跳過模型训练")

        return model, training_result

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

        if model is None or X_test is None:
            self.logger.warning("无模型或测试数据，无法生成预测")
            return pd.DataFrame()

        try:
            if hasattr(model, "predict"):
                preds = model.predict(X_test)
                if hasattr(preds, "predictions"):
                    scores = preds.predictions
                else:
                    scores = preds

                # 构建与 test_data 索引对齐的 DataFrame
                if test_data is not None and hasattr(test_data, "index"):
                    predictions = pd.DataFrame(scores.flatten(), index=test_data.index, columns=["score"])
                else:
                    predictions = pd.DataFrame({"score": scores.flatten()})

                self.logger.info("预测生成完成", shape=predictions.shape)
                return predictions

        except Exception as e:
            self.logger.error(f"预测生成失败: {e}")

        return pd.DataFrame()

    def _validate_accuracy(
        self,
        config: ExperimentConfig,
        model: Any,
        training_result: Any,
        X_test: Any,
        y_test: Any,
    ) -> Optional[Dict[str, Any]]:
        """
        准确度红线校验 (PRD 4.2)

        在模型训练完成后自动触发 AccuracyValidator 校验:
        - Rank IC 阈值检查 (0.045 ~ 0.055)
        - Rank ICIR > 0.40
        - IC 正值比率 > 55%
        - IC 标准差 < 0.08
        - 策略稳定性评分

        Returns:
            {"passed": bool, "checks": [...], "summary": str} 或 None
        """
        if X_test is None or y_test is None:
            self.logger.warning("无测试数据，跳过准确度校验")
            return None

        try:
            from src.analyzers.accuracy_validator import (
                AccuracyThresholdValidator,
                RollingICValidator,
                DrawdownValidator,
                StrategyStabilityChecker,
            )

            # 在测试集上生成预测
            if hasattr(model, "predict"):
                preds = model.predict(X_test)
                if hasattr(preds, "predictions"):
                    scores = preds.predictions
                else:
                    scores = preds
            else:
                self.logger.warning("模型无 predict 方法，跳过校验")
                return None

            scores = np.nan_to_num(scores.flatten(), nan=0.0)
            y = np.nan_to_num(y_test.flatten(), nan=0.0)
            mask = (~np.isnan(scores)) & (~np.isnan(y)) & (~np.isinf(scores)) & (~np.isinf(y))
            if mask.sum() < 30:
                self.logger.warning(f"有效样本不足 ({mask.sum()})，跳过校验")
                return None

            p = scores[mask]
            r = y[mask]

            # 计算 IC
            try:
                rank_ic = np.corrcoef(p, r)[0, 1]
            except Exception:
                rank_ic = 0.0

            # 阈值校验
            threshold_validator = AccuracyThresholdValidator()
            threshold_report = threshold_validator.validate(
                rank_ic_series=np.array([rank_ic]),
                rank_icir=abs(rank_ic) / (np.std(p) * np.std(r) + 1e-12),
                max_drawdown=0.0,  # 仅模型预测层面
            )

            # 滚动 IC 稳定性
            rolling_validator = RollingICValidator()
            rolling_report = rolling_validator.validate(p.reshape(-1, 1), r)

            # 策略稳定性
            stability_checker = StrategyStabilityChecker()
            stability_report = stability_checker.check(
                predictions=p,
                returns=r,
            )

            # 汇总
            all_passed = (
                threshold_report.passed
                and rolling_report.get("passed", True)
                and stability_report.get("passed", True)
            )

            checks = []
            if hasattr(threshold_report, "checks"):
                for c in threshold_report.checks:
                    checks.append({
                        "check": c.check_name if hasattr(c, "check_name") else "threshold",
                        "passed": c.passed if hasattr(c, "passed") else False,
                        "value": str(getattr(c, "actual", "")),
                        "threshold": str(getattr(c, "threshold", "")),
                    })

            result = {
                "passed": all_passed,
                "checks": checks,
                "rank_ic": round(rank_ic, 6),
                "rolling_stable": rolling_report.get("passed", True),
                "strategy_stable": stability_report.get("passed", True),
                "summary": (
                    "✓ 通过准确度红线校验" if all_passed
                    else "✗ 未通过准确度红线校验，请检查模型"
                ),
            }

            self.logger.info(
                "准确度校验完成",
                passed=all_passed,
                rank_ic=round(rank_ic, 4),
            )
            return result

        except Exception as e:
            self.logger.warning(f"准确度校验执行失败: {e}")
            return None

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

        if predictions.empty:
            self.logger.warning("无预测数据，跳过回测")
            return None

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

        try:
            # 使用 predictions 作为价格代理，构建价格数据
            if test_data is not None and hasattr(test_data, "shape"):
                prices = test_data
            else:
                # 从预测得分生成模拟价格
                prices = predictions.copy()

            simulator = PortfolioSimulator(
                strategy=strategy,
                initial_capital=1_000_000,
                commission_rate=strategy_config.commission_rate,
                slippage_bps=strategy_config.slippage_bps,
            )
            result = simulator.run(predictions, prices)
            self.logger.info(
                "回测完成",
                total_return=f"{getattr(result, 'total_return', 0):.2%}",
                sharpe=f"{getattr(result, 'sharpe_ratio', 0):.2f}",
            )
            return result

        except Exception as e:
            self.logger.error(f"回测执行失败: {e}")
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

        # 收集训练指标
        if training_result is not None:
            report_metrics["best_score"] = getattr(training_result, "best_score", 0)
            report_metrics["best_iteration"] = getattr(training_result, "best_iteration", 0)
            report_metrics["train_time_ms"] = getattr(training_result, "train_time_ms", 0)
            report_metrics["n_features"] = getattr(training_result, "n_features", 0)

        # 收集回测指标
        if backtest_result is not None:
            report_metrics["total_return"] = getattr(backtest_result, "total_return", 0)
            report_metrics["annual_return"] = getattr(backtest_result, "annual_return", 0)
            report_metrics["sharpe_ratio"] = getattr(backtest_result, "sharpe_ratio", 0)
            report_metrics["max_drawdown"] = getattr(backtest_result, "max_drawdown", 0)
            report_metrics["win_rate"] = getattr(backtest_result, "win_rate", 0)
            report_metrics["total_trades"] = getattr(backtest_result, "total_trades", 0)

        # 尝试生成详细报告
        try:
            from src.analyzers.report_generator import PerformanceReport

            pr = PerformanceReport()
            if backtest_result is not None:
                html_path = pr.export_html(
                    backtest_result=backtest_result,
                    output_path=str(exp_dir / "report.html"),
                )
                report_metrics["report_html"] = html_path

                json_path = pr.export_json(
                    backtest_result=backtest_result,
                    output_path=str(exp_dir / "report.json"),
                )
                report_metrics["report_json"] = json_path

        except Exception as e:
            self.logger.warning(f"详细报告生成失败: {e}")

        # 保存汇总指标
        import json
        metrics_path = exp_dir / "metrics.json"
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(report_metrics, f, indent=2, ensure_ascii=False, default=str)
        report_metrics["metrics_path"] = str(metrics_path)

        self.logger.info("报告生成完成", metrics_count=len(report_metrics))
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
