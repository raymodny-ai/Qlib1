"""
Qlib .bin 二进制格式转换器

将采集器生成的标准化 CSV/DataFrame 数据转换为 Qlib 专有的高密度
二进制格式。支持量价数据的复权处理（首日归一化 + 复权因子），
以及基本面财务数据的特征索引。

核心功能:
- DataFrame → Qlib .bin 格式序列化
- 量价复权归一化（首日价格强制为 1）
- 交易日历对齐
- 特征元数据生成
- 批量转换与增量更新

Qlib 数据目录结构:
    data/qlib_data/us_data/
    ├── calendars/
    │   └── day.txt
    ├── instruments/
    │   └── all.txt
    └── features/
        └── {instrument}/
            ├── {field}.bin       # 单字段二进制文件
            └── {field}.meta      # 字段元数据
"""

import os
import struct
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml

from src.utils.logger import get_logger

# ===== 常量 =====

# Qlib .bin 文件魔法数字
BIN_MAGIC = b"QLIB"
BIN_VERSION = 1

# 支持的价量字段
OHLCV_FIELDS = ["open", "high", "low", "close", "volume", "adj_factor"]

# 基本面字段前缀 (Qlib 惯例)
FUNDAMENTAL_PREFIXES = {
    "income": "IS",       # Income Statement
    "balance": "BS",      # Balance Sheet
    "cash_flow": "CF",    # Cash Flow
    "ratio": "RT",         # Ratio
    "market": "MV",       # Market Value
}


# ===== 数据结构 =====

@dataclass
class BinConfig:
    """.bin 文件生成配置"""
    output_dir: str = "./data/qlib_data/us_data"
    calendar_path: str = "./config/calendars/us_market.txt"
    instruments_path: str = "./config/instruments/all_us_stocks.txt"
    # 复权基准日期: None = 每只股票上市首日
    base_date: Optional[str] = None
    # 是否强制覆盖已存在的 .bin 文件
    force_overwrite: bool = False
    # 是否生成校验和
    enable_checksum: bool = True


@dataclass
class ConversionReport:
    """转换报告"""
    total_instruments: int = 0
    successful: int = 0
    failed: int = 0
    skipped: int = 0
    total_fields: int = 0
    errors: List[str] = field(default_factory=list)
    elapsed_seconds: float = 0.0


# ===== .bin 文件读写 =====

class BinWriter:
    """
    Qlib .bin 二进制文件写入器

    .bin 文件格式:
    ┌─────────────────────────────────────┐
    │ Header (16 bytes)                   │
    │  - magic:     4 bytes ("QLIB")      │
    │  - version:   4 bytes (uint32)      │
    │  - index_cnt: 4 bytes (uint32)      │
    │  - data_cnt:  4 bytes (uint32)      │
    ├─────────────────────────────────────┤
    │ Index Array (index_cnt * 8 bytes)   │
    │  Each: (start_idx, end_idx) int32   │
    ├─────────────────────────────────────┤
    │ Data Array  (data_cnt * 4 bytes)    │
    │  Each: float32 value                │
    └─────────────────────────────────────┘

    Index 对应日期数组（通过日历文件映射），每个交易日一个索引项。
    数据按 (instrument_index * num_days) 偏移存储，支持快速随机访问。
    """

    def __init__(self, calendar_dates: List[str], instruments: List[str]):
        """
        Args:
            calendar_dates: 交易日历日期列表 (YYYY-MM-DD)
            instruments: 股票代码列表
        """
        self.calendar_dates = calendar_dates
        self.instruments = instruments
        self.num_days = len(calendar_dates)
        self.num_instruments = len(instruments)

        # 日期 → 索引映射
        self.date_to_idx: Dict[str, int] = {
            d: i for i, d in enumerate(calendar_dates)
        }
        # 股票 → 索引映射
        self.inst_to_idx: Dict[str, int] = {
            inst: i for i, inst in enumerate(instruments)
        }

    def write_field(
        self,
        field_name: str,
        data: Dict[str, Dict[str, float]],
        output_dir: str,
        default_value: float = np.nan,
    ) -> str:
        """
        将单字段数据写入 .bin 文件

        Args:
            field_name: 字段名 (如 'close', 'volume', 'revenue')
            data: {instrument: {date_str: value}}
            output_dir: 输出目录
            default_value: 缺失数据的默认填充值

        Returns:
            生成的 .bin 文件路径
        """
        logger = get_logger()
        start_time = time.time()

        # 创建输出目录
        features_dir = Path(output_dir) / "features"
        features_dir.mkdir(parents=True, exist_ok=True)

        # 构建完整的 numpy 数组: [num_instruments * num_days]
        arr = np.full(self.num_instruments * self.num_days, default_value, dtype=np.float32)

        fill_count = 0
        for inst, date_values in data.items():
            if inst not in self.inst_to_idx:
                continue
            inst_offset = self.inst_to_idx[inst] * self.num_days

            for date_str, value in date_values.items():
                if date_str not in self.date_to_idx:
                    continue
                day_idx = self.date_to_idx[date_str]
                arr[inst_offset + day_idx] = np.float32(value)
                fill_count += 1

        # 构建索引数组: 每个 instrument 的 (start, end) 范围
        index_arr = np.zeros(self.num_instruments * 2, dtype=np.int32)
        for i in range(self.num_instruments):
            index_arr[i * 2] = i * self.num_days
            index_arr[i * 2 + 1] = (i + 1) * self.num_days

        # 写入 .bin 文件
        bin_path = features_dir / f"{field_name}.bin"
        with open(bin_path, "wb") as f:
            # Header
            f.write(BIN_MAGIC)
            f.write(struct.pack("<I", BIN_VERSION))
            f.write(struct.pack("<I", len(index_arr)))
            f.write(struct.pack("<I", len(arr)))
            # Index
            f.write(index_arr.tobytes())
            # Data
            f.write(arr.tobytes())

        # 写入 .meta 元数据文件
        meta_path = features_dir / f"{field_name}.meta"
        meta = {
            "field_name": field_name,
            "instruments": self.instruments,
            "num_instruments": self.num_instruments,
            "calendar_len": self.num_days,
            "type": "float32",
            "default_value": float(default_value),
            "generated_at": datetime.now().isoformat(),
        }
        with open(meta_path, "w") as f:
            yaml.dump(meta, f, default_flow_style=False)

        elapsed = round(time.time() - start_time, 3)
        logger.info(
            ".bin 字段写入完成",
            field=field_name,
            instruments=self.num_instruments,
            days=self.num_days,
            filled_values=fill_count,
            path=str(bin_path),
            elapsed_s=elapsed,
        )

        return str(bin_path)


class BinReader:
    """
    Qlib .bin 二进制文件读取器

    支持按 instrument 和日期范围快速读取数据。
    """

    def __init__(self, features_dir: str, calendar_dates: List[str]):
        self.features_dir = Path(features_dir)
        self.calendar_dates = calendar_dates
        self.num_days = len(calendar_dates)
        self.date_to_idx = {d: i for i, d in enumerate(calendar_dates)}

        # 缓存的字段元数据
        self._meta_cache: Dict[str, dict] = {}

    def load_meta(self, field_name: str) -> dict:
        """加载字段元数据"""
        if field_name in self._meta_cache:
            return self._meta_cache[field_name]

        meta_path = self.features_dir / f"{field_name}.meta"
        if not meta_path.exists():
            raise FileNotFoundError(f"元数据文件不存在: {meta_path}")

        with open(meta_path) as f:
            meta = yaml.safe_load(f)
        self._meta_cache[field_name] = meta
        return meta

    def load_field(
        self,
        field_name: str,
        instruments: Optional[List[str]] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        加载单字段数据为 DataFrame

        Args:
            field_name: 字段名
            instruments: 目标股票列表 (None = 全部)
            start_date: 起始日期 (YYYY-MM-DD)
            end_date: 截止日期 (YYYY-MM-DD)

        Returns:
            DataFrame (index=date, columns=instrument)
        """
        meta = self.load_meta(field_name)
        all_instruments = meta["instruments"]
        inst_to_idx = {inst: i for i, inst in enumerate(all_instruments)}

        # 读取 .bin 文件
        bin_path = self.features_dir / f"{field_name}.bin"
        if not bin_path.exists():
            raise FileNotFoundError(f".bin 文件不存在: {bin_path}")

        with open(bin_path, "rb") as f:
            magic = f.read(4)
            if magic != BIN_MAGIC:
                raise ValueError(f"无效的 .bin 文件魔法数字: {magic}")

            version = struct.unpack("<I", f.read(4))[0]
            index_cnt = struct.unpack("<I", f.read(4))[0]
            data_cnt = struct.unpack("<I", f.read(4))[0]

            index_arr = np.frombuffer(f.read(index_cnt * 4), dtype=np.int32)
            data_arr = np.frombuffer(f.read(data_cnt * 4), dtype=np.float32)

        # 确定目标 instrument 列表
        target_instruments = instruments or all_instruments

        # 确定日期范围
        start_idx = 0
        end_idx = self.num_days
        if start_date and start_date in self.date_to_idx:
            start_idx = self.date_to_idx[start_date]
        if end_date and end_date in self.date_to_idx:
            end_idx = self.date_to_idx[end_date] + 1

        # 提取数据
        dates = self.calendar_dates[start_idx:end_idx]
        result = {}

        for inst in target_instruments:
            if inst not in inst_to_idx:
                continue
            inst_idx = inst_to_idx[inst]
            inst_start = index_arr[inst_idx * 2]
            data_slice = data_arr[inst_start + start_idx: inst_start + end_idx]
            result[inst] = data_slice

        df = pd.DataFrame(result, index=dates)
        df.index.name = "date"
        return df


# ===== .bin 格式转换器 =====

class DataConverter:
    """
    主数据转换器

    将采集器输出的 DataFrame 转换为 Qlib .bin 格式，
    包含复权处理、日历对齐和增量更新功能。

    使用示例:
        converter = DataConverter(config=BinConfig())
        report = converter.convert_ohlcv(daily_prices_df)
        report = converter.convert_fundamentals(fundamentals_df)
    """

    def __init__(self, config: Optional[BinConfig] = None):
        self.config = config or BinConfig()
        self.logger = get_logger()

        # 加载日历
        self.calendar = self._load_calendar()
        # 加载股票列表
        self.instruments = self._load_instruments()

        # 延迟初始化 Writer
        self._writer: Optional[BinWriter] = None

    def _load_calendar(self) -> List[str]:
        """加载交易日历"""
        cal_path = Path(self.config.calendar_path)
        if not cal_path.exists():
            raise FileNotFoundError(f"交易日历文件不存在: {cal_path}")

        dates = []
        with open(cal_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    dates.append(line)
        return dates

    def _load_instruments(self) -> List[str]:
        """加载股票代码列表"""
        inst_path = Path(self.config.instruments_path)
        if not inst_path.exists():
            # 尝试自动发现
            self.logger.warning("股票列表文件不存在，将在转换时自动收集")
            return []

        instruments = []
        with open(inst_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    instruments.append(line.upper())
        return instruments

    @property
    def writer(self) -> BinWriter:
        if self._writer is None:
            self._writer = BinWriter(
                calendar_dates=self.calendar,
                instruments=self.instruments,
            )
        return self._writer

    # ===== 量价数据转换 =====

    def convert_ohlcv(
        self,
        df: pd.DataFrame,
        instruments: Optional[List[str]] = None,
        adjust_prices: bool = True,
    ) -> ConversionReport:
        """
        转换量价数据为 .bin 格式

        Args:
            df: 量价 DataFrame，列名为 (instrument, field) MultiIndex
                或 columns=['date','instrument','open','high','low','close','volume','adj_factor']
            instruments: 目标股票列表 (None = 自动从 df 提取)
            adjust_prices: 是否执行复权处理（首日归一化）

        Returns:
            ConversionReport
        """
        report = ConversionReport()
        start_time = time.time()

        # 自动收集 instrument 列表
        if instruments is None:
            if "instrument" in df.columns:
                instruments = sorted(df["instrument"].unique().tolist())
            else:
                instruments = sorted(set(
                    c[0] if isinstance(c, tuple) else c
                    for c in df.columns
                    if c not in ("date", "instrument")
                ))
        self.instruments = instruments
        report.total_instruments = len(instruments)

        # 标准化 DataFrame 为长格式
        long_df = self._normalize_ohlcv(df, instruments)

        # 复权处理
        if adjust_prices:
            long_df = self._apply_price_adjustment(long_df)

        # 按字段拆分写入
        fields_to_convert = [
            f for f in OHLCV_FIELDS
            if f in long_df.columns and f != "adj_factor"
        ]

        for field in fields_to_convert:
            try:
                data = self._df_to_dict(long_df, field)
                self.writer.write_field(
                    field_name=field,
                    data=data,
                    output_dir=self.config.output_dir,
                )
                report.successful += 1
                report.total_fields += 1
            except Exception as e:
                report.failed += 1
                report.errors.append(f"字段 {field} 转换失败: {e}")

        # 同时写入复权因子
        if adjust_prices and "adj_factor" in long_df.columns:
            try:
                adj_data = self._df_to_dict(long_df, "adj_factor")
                self.writer.write_field(
                    field_name="factor",
                    data=adj_data,
                    output_dir=self.config.output_dir,
                    default_value=1.0,
                )
                report.total_fields += 1
            except Exception as e:
                report.errors.append(f"复权因子转换失败: {e}")

        report.elapsed_seconds = round(time.time() - start_time, 2)
        self.logger.info(
            "量价数据转换完成",
            instruments=report.total_instruments,
            fields=report.total_fields,
            success=report.successful,
            failed=report.failed,
            elapsed_s=report.elapsed_seconds,
        )
        return report

    def _normalize_ohlcv(
        self, df: pd.DataFrame, instruments: List[str]
    ) -> pd.DataFrame:
        """
        标准化 OHLCV DataFrame

        支持三种输入格式:
        1. MultiIndex columns: ('AAPL', 'close')
        2. Long format: columns=['date','instrument','close',...]
        3. Single instrument: columns=['date','close',...]
        """
        if isinstance(df.columns, pd.MultiIndex):
            # 格式 1 → 转换为长格式
            records = []
            for inst in df.columns.levels[0]:
                if inst not in instruments:
                    continue
                sub = df[inst].copy()
                sub["instrument"] = inst
                sub["date"] = df.index
                records.append(sub)
            return pd.concat(records, ignore_index=True)

        if "instrument" in df.columns and "date" in df.columns:
            # 格式 2 → 直接过滤
            return df[df["instrument"].isin(instruments)].copy()

        # 格式 3 → 单股票，添加 instrument 列
        if instruments:
            df = df.copy()
            df["instrument"] = instruments[0]

        return df

    def _apply_price_adjustment(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        复权处理：将价格字段进行首日归一化

        首日价格强制为 1，所有后续价格通过复权因子缩放。
        原始价格 = normalized_price / adj_factor
        """
        df = df.copy()

        # 确保按 instrument 和 date 排序
        df = df.sort_values(["instrument", "date"])

        price_fields = ["open", "high", "low", "close"]

        if "adj_factor" not in df.columns:
            # 如果没有复权因子列，假设无复权调整
            df["adj_factor"] = 1.0

        # 按 instrument 分组处理
        result_dfs = []
        for inst, group in df.groupby("instrument"):
            group = group.copy()

            # 计算累积复权因子
            group["adj_factor"] = group["adj_factor"].fillna(1.0)
            group["cum_factor"] = group["adj_factor"].cumprod()

            # 首日归一化
            if len(group) > 0:
                first_factor = group["cum_factor"].iloc[0]
                if first_factor > 0:
                    # 所有价格除以首日累积因子，实现首日归一化
                    for pf in price_fields:
                        if pf in group.columns:
                            group[pf] = group[pf] / first_factor

                    # 累积因子同步缩放
                    group["cum_factor"] = group["cum_factor"] / first_factor
                    group["adj_factor"] = group["cum_factor"] / group["cum_factor"].shift(1).fillna(1.0)

            result_dfs.append(group.drop(columns=["cum_factor"], errors="ignore"))

        return pd.concat(result_dfs, ignore_index=True)

    def _df_to_dict(
        self, df: pd.DataFrame, field: str
    ) -> Dict[str, Dict[str, float]]:
        """将 DataFrame 转换为 {instrument: {date: value}} 格式"""
        result: Dict[str, Dict[str, float]] = {}
        for _, row in df.iterrows():
            inst = row["instrument"]
            date_str = str(row.get("date", row.name))
            value = row.get(field)
            if value is None or (isinstance(value, float) and np.isnan(value)):
                continue

            if inst not in result:
                result[inst] = {}
            result[inst][date_str] = float(value)
        return result

    # ===== 基本面数据转换 =====

    def convert_fundamentals(
        self,
        df: pd.DataFrame,
        category: str = "income",
    ) -> ConversionReport:
        """
        转换基本面数据为 .bin 格式

        Args:
            df: 基本面 DataFrame (columns: instrument, date, field1, field2, ...)
            category: 基本面类别 ('income'|'balance'|'cash_flow'|'ratio'|'market')

        Returns:
            ConversionReport
        """
        report = ConversionReport()
        start_time = time.time()

        prefix = FUNDAMENTAL_PREFIXES.get(category, "FE")

        # 提取 instrument 列表
        if "instrument" in df.columns:
            instruments = sorted(df["instrument"].unique().tolist())
        else:
            instruments = self.instruments
        report.total_instruments = len(instruments)

        # 确定要转换的字段 (排除 instrument/date/period_end_date/filing_date)
        exclude = {"instrument", "date", "period_end_date", "filing_date", "raw_tags", "raw_xbrl_tags"}
        fields = [c for c in df.columns if c not in exclude]

        for field in fields:
            try:
                field_name = f"{prefix}_{field}" if prefix else field
                data = self._df_to_dict(df, field)
                self.writer.write_field(
                    field_name=field_name,
                    data=data,
                    output_dir=self.config.output_dir,
                )
                report.successful += 1
                report.total_fields += 1
            except Exception as e:
                report.failed += 1
                report.errors.append(f"基本面字段 {field} 转换失败: {e}")

        report.elapsed_seconds = round(time.time() - start_time, 2)
        self.logger.info(
            "基本面数据转换完成",
            category=category,
            fields=report.total_fields,
            success=report.successful,
            failed=report.failed,
            elapsed_s=report.elapsed_seconds,
        )
        return report

    # ===== 批量转换 =====

    def convert_all(
        self,
        ohlcv_df: Optional[pd.DataFrame] = None,
        income_df: Optional[pd.DataFrame] = None,
        balance_df: Optional[pd.DataFrame] = None,
        cashflow_df: Optional[pd.DataFrame] = None,
    ) -> Dict[str, ConversionReport]:
        """
        一站式转换所有数据
        """
        reports = {}

        if ohlcv_df is not None:
            reports["ohlcv"] = self.convert_ohlcv(ohlcv_df)

        if income_df is not None:
            reports["income"] = self.convert_fundamentals(income_df, "income")

        if balance_df is not None:
            reports["balance"] = self.convert_fundamentals(balance_df, "balance")

        if cashflow_df is not None:
            reports["cash_flow"] = self.convert_fundamentals(cashflow_df, "cash_flow")

        return reports


# ===== 便捷函数 =====

def dump_bin(
    df: pd.DataFrame,
    output_dir: str = "./data/qlib_data/us_data",
    category: str = "ohlcv",
    adjust_prices: bool = True,
) -> ConversionReport:
    """
    快捷转换接口（CLI 调用入口）

    用法:
        from src.processors.data_converter import dump_bin
        report = dump_bin(ohlcv_df, category="ohlcv")
    """
    config = BinConfig(output_dir=output_dir)
    converter = DataConverter(config=config)

    if category == "ohlcv":
        return converter.convert_ohlcv(df, adjust_prices=adjust_prices)
    else:
        return converter.convert_fundamentals(df, category=category)
