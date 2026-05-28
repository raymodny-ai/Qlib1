"""
模型训练流水线 CLI 与编排模块

从配置文件加载模型参数，执行训练→验证→保存全流程。
包装 src.analyzers.ml_pipeline 中各模型实现。

用法:
    python -m src.workflow.training_pipeline --model lightgbm --config config/qlib_config.yaml
    python -m src.workflow.training_pipeline --model all --gpu 0
"""

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import yaml

from src.analyzers.ml_pipeline import (
    BaseForecastModel,
    LightGBMModel,
    XGBoostModel,
    MLPipeline,
    TrainingResult,
)
from src.utils.logger import get_logger


MODEL_REGISTRY = {
    "lightgbm": LightGBMModel,
    "xgboost": XGBoostModel,
    # Phase 2 will add: "adarnn", "tabnet", "double_ensemble"
}


def load_data_splits(
    data_dir: str = "./data/qlib_data/us_data",
    train_start: str = "2015-01-01",
    train_end: str = "2019-12-31",
    valid_start: str = "2020-01-01",
    valid_end: str = "2020-12-31",
    test_start: str = "2021-01-01",
    test_end: str = "2023-12-31",
    label_col: str = "label",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    从特征数据中加载训练/验证/测试集。

    实际生产环境应从 DataServer 或 PIT 数据库加载，
    当前为简化实现，支持从 parquet 文件读取。
    """
    logger = get_logger(__name__)

    features_path = Path(data_dir) / "features.parquet"
    if not features_path.exists():
        logger.warning(f"特征文件不存在: {features_path}，使用空数据占位")
        n_features = 50
        n_samples = 1000
        X = np.random.randn(n_samples, n_features).astype(np.float32)
        y = np.random.randn(n_samples).astype(np.float32)
        return (
            X[:600], y[:600],
            X[600:800], y[600:800],
            X[800:], y[800:],
        )

    df = pd.read_parquet(features_path)

    # 按时间切分
    if "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime"])
        df = df.sort_values("datetime")

        train_mask = (df["datetime"] >= train_start) & (df["datetime"] <= train_end)
        valid_mask = (df["datetime"] >= valid_start) & (df["datetime"] <= valid_end)
        test_mask = (df["datetime"] >= test_start) & (df["datetime"] <= test_end)
    elif "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date")

        train_mask = (df["date"] >= train_start) & (df["date"] <= train_end)
        valid_mask = (df["date"] >= valid_start) & (df["date"] <= valid_end)
        test_mask = (df["date"] >= test_start) & (df["date"] <= test_end)
    else:
        # 无日期列时简单地按行切分
        n = len(df)
        train_mask = np.arange(n) < int(n * 0.6)
        valid_mask = (np.arange(n) >= int(n * 0.6)) & (np.arange(n) < int(n * 0.8))
        test_mask = np.arange(n) >= int(n * 0.8)

    # 分离特征和标签
    feature_cols = [c for c in df.columns if c not in ("datetime", "date", "instrument", label_col)]
    X_cols = [c for c in feature_cols if df[c].dtype in ("float64", "float32", "int64", "int32")]

    X_train = df.loc[train_mask, X_cols].fillna(0).values.astype(np.float32)
    y_train = df.loc[train_mask, label_col].fillna(0).values.astype(np.float32) if label_col in df.columns else np.zeros(X_train.shape[0])

    X_valid = df.loc[valid_mask, X_cols].fillna(0).values.astype(np.float32)
    y_valid = df.loc[valid_mask, label_col].fillna(0).values.astype(np.float32) if label_col in df.columns else np.zeros(X_valid.shape[0])

    X_test = df.loc[test_mask, X_cols].fillna(0).values.astype(np.float32)
    y_test = df.loc[test_mask, label_col].fillna(0).values.astype(np.float32) if label_col in df.columns else np.zeros(X_test.shape[0])

    logger.info(
        "数据加载完成",
        train=(X_train.shape, y_train.shape),
        valid=(X_valid.shape, y_valid.shape) if X_valid.size > 0 else "N/A",
        test=(X_test.shape, y_test.shape),
    )

    return X_train, y_train, X_valid, y_valid, X_test, y_test


def run_training(
    model_type: str = "lightgbm",
    config_path: Optional[str] = None,
    data_dir: str = "./data/qlib_data/us_data",
    output_dir: str = "./models",
    gpu_id: int = -1,
) -> Dict[str, Any]:
    """
    执行模型训练流水线

    Args:
        model_type: 模型类型 (lightgbm | xgboost | all)
        config_path: YAML 配置文件路径
        data_dir: Qlib 数据目录
        output_dir: 模型输出目录
        gpu_id: GPU 设备 ID (-1 = CPU)

    Returns:
        训练结果摘要
    """
    logger = get_logger(__name__)

    # 加载配置
    model_params = {}
    if config_path:
        config_file = Path(config_path)
        if config_file.exists():
            with open(config_file, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)
            models_cfg = config.get("models", {})
            if model_type in models_cfg:
                model_params = models_cfg[model_type]
            logger.info("已加载模型配置", path=str(config_file), params=model_params)

    # 加载数据
    X_train, y_train, X_valid, y_valid, X_test, y_test = load_data_splits(data_dir)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    results = {}

    model_types = [model_type]
    if model_type == "all":
        model_types = list(MODEL_REGISTRY.keys())

    for mtype in model_types:
        model_cls = MODEL_REGISTRY.get(mtype)
        if model_cls is None:
            logger.warning(f"未知模型类型: {mtype}，跳过")
            continue

        logger.info(f"开始训练: {mtype}")

        # 合并默认参数与配置参数
        model = model_cls(**model_params)
        pipeline = MLPipeline(model)

        training_result = pipeline.fit(
            X_train, y_train,
            X_valid if X_valid.size > 0 else None,
            y_valid if y_valid.size > 0 else None,
        )

        # 预测
        predictions = pipeline.predict(X_test)
        ic = np.corrcoef(predictions, y_test)[0, 1] if y_test.std() > 0 else 0.0

        # 保存模型
        model_path = output_path / f"{mtype}_model.pkl"
        pipeline.save(str(model_path))

        results[mtype] = {
            "model": mtype,
            "best_score": training_result.best_score if hasattr(training_result, "best_score") else 0,
            "best_iteration": training_result.best_iteration if hasattr(training_result, "best_iteration") else 0,
            "train_time_ms": training_result.train_time_ms if hasattr(training_result, "train_time_ms") else 0,
            "ic": round(ic, 6),
            "model_path": str(model_path),
        }

        logger.info(
            f"训练完成: {mtype}",
            best_score=round(results[mtype]["best_score"], 4) if isinstance(results[mtype]["best_score"], (int, float)) else results[mtype]["best_score"],
            ic=round(ic, 4),
            path=str(model_path),
        )

    return results


def main():
    parser = argparse.ArgumentParser(
        description="模型训练流水线 (Training Pipeline)",
    )
    parser.add_argument(
        "--model", "-m",
        default="lightgbm",
        choices=["lightgbm", "xgboost", "all"],
        help="模型类型",
    )
    parser.add_argument(
        "--config", "-c",
        default="config/qlib_config.yaml",
        help="配置文件路径",
    )
    parser.add_argument(
        "--data-dir", "-d",
        default="./data/qlib_data/us_data",
        help="Qlib 数据目录",
    )
    parser.add_argument(
        "--output-dir", "-o",
        default="./models",
        help="模型输出目录",
    )
    parser.add_argument(
        "--gpu", "-g",
        type=int,
        default=-1,
        help="GPU 设备 ID (-1 = CPU)",
    )

    args = parser.parse_args()

    logger = get_logger(__name__)

    try:
        results = run_training(
            model_type=args.model,
            config_path=args.config,
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            gpu_id=args.gpu,
        )

        print(f"\nTraining Summary ({len(results)} models):")
        for mtype, r in results.items():
            print(f"  [{mtype}] IC: {r['ic']:.4f} | Model: {r['model_path']}")

    except Exception as e:
        logger.error(f"训练流水线执行失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
