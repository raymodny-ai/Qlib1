"""
Qlib .bin 二进制格式转换 CLI

将 CSV/Parquet 原始数据转换为 Qlib 高密度二进制格式 (.bin)。
包装 src.processors.data_converter.DataConverter。

用法:
    python scripts/dump_bin.py --input-dir ./data/raw --output-dir ./data/qlib_data/us_data
    python scripts/dump_bin.py --input-dir ./data/raw --category ohlcv --adjust-prices
    python scripts/dump_bin.py --input-dir ./data/raw --category fundamentals --full
"""

import argparse
import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from src.processors.data_converter import BinConfig, DataConverter, dump_bin
from src.utils.logger import get_logger


def main():
    parser = argparse.ArgumentParser(
        description="Qlib .bin 二进制格式转换工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--input-dir", "-i",
        default="./data/raw",
        help="原始 CSV/Parquet 数据目录 (默认: ./data/raw)",
    )
    parser.add_argument(
        "--output-dir", "-o",
        default="./data/qlib_data/us_data",
        help="Qlib .bin 输出目录 (默认: ./data/qlib_data/us_data)",
    )
    parser.add_argument(
        "--category", "-c",
        default="ohlcv",
        choices=["ohlcv", "income", "balance", "cash_flow", "fundamentals"],
        help="数据类别 (默认: ohlcv)",
    )
    parser.add_argument(
        "--input-file", "-f",
        default=None,
        help="指定输入文件路径 (支持 .csv / .parquet)",
    )
    parser.add_argument(
        "--adjust-prices",
        action="store_true",
        default=True,
        help="启用复权处理 (默认: True)",
    )
    parser.add_argument(
        "--no-adjust-prices",
        action="store_false",
        dest="adjust_prices",
        help="禁用复权处理",
    )
    parser.add_argument(
        "--incremental",
        action="store_true",
        default=True,
        help="增量模式，仅转换新数据 (默认)",
    )
    parser.add_argument(
        "--full",
        action="store_false",
        dest="incremental",
        help="全量重建模式",
    )
    parser.add_argument(
        "--tickers",
        default=None,
        help="股票代码文件路径，每行一个代码 (可选)",
    )

    args = parser.parse_args()

    logger = get_logger(__name__)
    logger.info("启动 .bin 格式转换", category=args.category, output=args.output_dir)

    # Case 1: 指定了单个输入文件
    if args.input_file:
        input_path = Path(args.input_file)
        if not input_path.exists():
            logger.error(f"输入文件不存在: {input_path}")
            sys.exit(1)

        if input_path.suffix == ".csv":
            df = pd.read_csv(input_path)
        elif input_path.suffix == ".parquet":
            df = pd.read_parquet(input_path)
        else:
            logger.error(f"不支持的文件格式: {input_path.suffix}")
            sys.exit(1)

        report = dump_bin(
            df=df,
            output_dir=args.output_dir,
            category=args.category,
            adjust_prices=args.adjust_prices,
        )
        logger.info(
            "转换完成",
            category=report.category,
            instruments=report.instruments_written,
            fields=report.fields_written,
            records=report.records_written,
            duration_s=round(report.duration_seconds, 2),
        )
        print(f"[OK] {report.category}: {report.instruments_written} instruments, "
              f"{report.fields_written} fields, {report.records_written} records "
              f"({report.duration_seconds:.1f}s)")
        return

    # Case 2: 目录批量转换
    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        logger.error(f"输入目录不存在: {input_dir}")
        sys.exit(1)

    config = BinConfig(output_dir=args.output_dir)
    converter = DataConverter(config=config)

    # 发现数据文件
    tickers = None
    if args.tickers:
        with open(args.tickers, "r") as f:
            tickers = [line.strip() for line in f if line.strip()]

    results = {}

    # 自动检测: ohlcv/
    ohlcv_dir = input_dir / "ohlcv"
    if ohlcv_dir.exists():
        logger.info("检测到 OHLCV 数据", path=str(ohlcv_dir))
        dfs = []
        for f in sorted(ohlcv_dir.glob("*.csv")) + sorted(ohlcv_dir.glob("*.parquet")):
            if tickers and f.stem.upper() not in tickers:
                continue
            try:
                df = pd.read_csv(f) if f.suffix == ".csv" else pd.read_parquet(f)
                dfs.append(df)
            except Exception as e:
                logger.warning(f"跳过 {f.name}: {e}")

        if dfs:
            merged = pd.concat(dfs, ignore_index=True)
            report = converter.convert_ohlcv(merged, adjust_prices=args.adjust_prices)
            results["ohlcv"] = report
            print(f"[OK] ohlcv: {report.instruments_written} instruments, "
                  f"{report.records_written} records ({report.duration_seconds:.1f}s)")

    # 自动检测: fundamentals/
    for category in ["income", "balance", "cash_flow"]:
        cat_dir = input_dir / category
        if cat_dir.exists():
            logger.info(f"检测到 {category} 数据", path=str(cat_dir))
            dfs = []
            for f in sorted(cat_dir.glob("*.csv")) + sorted(cat_dir.glob("*.parquet")):
                if tickers and f.stem.upper() not in tickers:
                    continue
                try:
                    df = pd.read_csv(f) if f.suffix == ".csv" else pd.read_parquet(f)
                    dfs.append(df)
                except Exception as e:
                    logger.warning(f"跳过 {f.name}: {e}")

            if dfs:
                merged = pd.concat(dfs, ignore_index=True)
                report = converter.convert_fundamentals(merged, category=category)
                results[category] = report
                print(f"[OK] {category}: {report.instruments_written} instruments, "
                      f"{report.records_written} records ({report.duration_seconds:.1f}s)")

    if not results:
        logger.warning("未发现可转换的数据文件。请确保目录结构为:")
        logger.warning("  {input_dir}/ohlcv/*.csv")
        logger.warning("  {input_dir}/income/*.csv")
        logger.warning("  {input_dir}/balance/*.csv")
        logger.warning("  {input_dir}/cash_flow/*.csv")
        sys.exit(1)

    total_instruments = sum(r.instruments_written for r in results.values())
    total_records = sum(r.records_written for r in results.values())
    total_time = sum(r.duration_seconds for r in results.values())
    logger.info(
        "批量转换完成",
        categories=len(results),
        total_instruments=total_instruments,
        total_records=total_records,
        total_time_s=round(total_time, 2),
    )
    print(f"\nSummary: {len(results)} categories, {total_instruments} instruments, "
          f"{total_records} records ({total_time:.1f}s)")


if __name__ == "__main__":
    main()
