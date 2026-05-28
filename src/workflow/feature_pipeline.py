"""
特征工程流水线 CLI 与编排模块

将原始特征矩阵通过可配置的处理器链执行清洗与标准化。
包装 src.processors.feature_pipeline.FeaturePipeline。

用法:
    python -m src.workflow.feature_pipeline --config config/qlib_config.yaml
    python -m src.workflow.feature_pipeline --input data.parquet --output features.parquet
"""

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import yaml

from src.processors.feature_pipeline import FeaturePipeline
from src.processors.feature_pipeline import (
    RobusZScoreNorm as RobustZScoreNorm,
    Fillna,
    CSRankNorm,
    Winsorize,
    DropnaLabel,
)
from src.utils.logger import get_logger


PROCESSOR_MAP = {
    "robust_zscore": RobustZScoreNorm,
    "robustzscorenorm": RobustZScoreNorm,
    "fillna": Fillna,
    "csranknorm": CSRankNorm,
    "cs_rank_norm": CSRankNorm,
    "winsorize": Winsorize,
    "dropna_label": DropnaLabel,
    "dropnalabel": DropnaLabel,
}


def build_pipeline_from_config(config_path: str) -> FeaturePipeline:
    """从 YAML 配置文件构建特征处理管道"""
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # 支持两种配置格式:
    # 1. 顶层 processors: 列表
    # 2. data_handler.processors: 列表 (qlib_config.yaml 格式)
    processor_configs = config.get("processors") or config.get("data_handler", {}).get("processors", [])

    if not processor_configs:
        raise ValueError(f"配置文件中未找到 processors 定义: {config_path}")

    return FeaturePipeline(processor_configs)


def build_pipeline_from_list(processor_list: List[Dict[str, Any]]) -> FeaturePipeline:
    """从处理器配置列表构建管道"""
    return FeaturePipeline(processor_list)


def run_feature_pipeline(
    input_path: str,
    output_path: Optional[str] = None,
    config_path: Optional[str] = None,
    processor_list: Optional[List[Dict[str, Any]]] = None,
    label_col: str = "label",
) -> pd.DataFrame:
    """
    执行特征工程管道

    Args:
        input_path: 输入数据文件路径 (.csv / .parquet)
        output_path: 输出路径 (可选)
        config_path: YAML 配置文件路径
        processor_list: 处理器配置列表 (与 config_path 二选一)
        label_col: 标签列名

    Returns:
        处理后的 DataFrame
    """
    logger = get_logger(__name__)

    # 加载数据
    input_file = Path(input_path)
    if not input_file.exists():
        raise FileNotFoundError(f"输入文件不存在: {input_path}")

    if input_file.suffix == ".csv":
        df = pd.read_csv(input_file)
    elif input_file.suffix == ".parquet":
        df = pd.read_parquet(input_file)
    else:
        raise ValueError(f"不支持的文件格式: {input_file.suffix}")

    logger.info("数据加载完成", shape=df.shape, path=str(input_file))

    # 构建管道
    if config_path:
        pipeline = build_pipeline_from_config(config_path)
    elif processor_list:
        pipeline = build_pipeline_from_list(processor_list)
    else:
        # 默认处理器链
        pipeline = FeaturePipeline([
            {"type": "dropna_label", "label_col": label_col},
            {"type": "winsorize", "limits": [0.01, 0.99]},
            {"type": "fillna", "strategy": "cross_sectional"},
            {"type": "robust_zscore", "clip_range": 3.0},
            {"type": "cs_rank_norm"},
        ])

    logger.info("特征管道已构建", processors=pipeline.processor_names)

    # 执行处理
    result = pipeline.fit_transform(df)

    logger.info("特征处理完成", input_shape=df.shape, output_shape=result.shape)

    # 保存
    if output_path:
        out_file = Path(output_path)
        if out_file.suffix == ".csv":
            result.to_csv(output_path, index=False)
        elif out_file.suffix == ".parquet":
            result.to_parquet(output_path)
        else:
            result.to_parquet(f"{output_path}.parquet")
        logger.info("结果已保存", path=str(output_path))

    return result


def main():
    parser = argparse.ArgumentParser(
        description="特征工程流水线 (Feature Engineering Pipeline)",
    )
    parser.add_argument(
        "--config", "-c",
        default="config/qlib_config.yaml",
        help="YAML 配置文件路径",
    )
    parser.add_argument(
        "--input", "-i",
        default="./data/raw/features.parquet",
        help="输入数据文件路径",
    )
    parser.add_argument(
        "--output", "-o",
        default="./data/processed/features.parquet",
        help="输出文件路径",
    )
    parser.add_argument(
        "--label-col",
        default="label",
        help="标签列名 (默认: label)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="仅打印管道配置，不执行处理",
    )

    args = parser.parse_args()

    logger = get_logger(__name__)

    if args.dry_run:
        config_path = Path(args.config)
        if config_path.exists():
            pipeline = build_pipeline_from_config(str(config_path))
        else:
            pipeline = FeaturePipeline([
                {"type": "dropna_label", "label_col": args.label_col},
                {"type": "winsorize", "limits": [0.01, 0.99]},
                {"type": "fillna", "strategy": "cross_sectional"},
                {"type": "robust_zscore", "clip_range": 3.0},
            ])
        print(f"Processors ({len(pipeline.processors)}):")
        for p in pipeline.processors:
            print(f"  - {p.name}: {p.get_params()}")
        return

    try:
        result = run_feature_pipeline(
            input_path=args.input,
            output_path=args.output,
            config_path=args.config,
            label_col=args.label_col,
        )
        print(f"Feature pipeline complete: {result.shape[0]} rows, {result.shape[1]} cols")
    except Exception as e:
        logger.error(f"特征管道执行失败: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
