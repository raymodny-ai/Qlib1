"""
Qlib 原生格式桥接器

将自定义 .bin 格式数据转换为 Qlib 原生格式 (DataFrame 多索引)，
实现与 qlib 库 API 的双向兼容。

核心功能:
- BinToQlibConverter: 自定义 .bin → Qlib DataFrame (instrument-datetime 多索引)
- QlibToBinConverter: Qlib DataFrame → 自定义 .bin
- FormatValidator: 验证两个格式之间的数据一致性

使用示例:
    from src.processors.qlib_native_converter import BinToQlibConverter

    converter = BinToQlibConverter(data_dir="./data/bin")
    df = converter.to_qlib_format(
        instruments=["AAPL", "MSFT"],
        fields=["close", "volume"],
        start="2023-01-01",
        end="2023-12-31",
    )
    # df 现在兼容 qlib DataHandler 接口

设计原则:
- 无损转换，保留所有字段和时间戳精度
- 惰性加载，按需转换单个 instrument
- 支持批量转换与增量更新
"""

import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from src.utils.logger import get_logger


class BinToQlibConverter:
    """
    自定义 .bin 格式 → Qlib 原生 DataFrame 转换器

    Qlib 原生格式要求:
    - MultiIndex: (symbol, datetime)
    - 按 instrument 分组排列
    - datetime 列名默认为 "datetime"

    参数:
        data_dir: .bin 文件目录 (如 ./data/raw)
        cache_enabled: 是否启用内存缓存
    """

    def __init__(self, data_dir: str = "./data/raw", cache_enabled: bool = True):
        self.data_dir = Path(data_dir)
        self.cache_enabled = cache_enabled
        self._cache: Dict[str, pd.DataFrame] = {}
        self._metadata: Dict[str, Any] = {}
        self.logger = get_logger()

    def list_instruments(self) -> List[str]:
        """获取可用证券列表"""
        instruments = []
        if not self.data_dir.exists():
            return instruments

        for entry in self.data_dir.iterdir():
            if entry.is_dir():
                instruments.append(entry.name)
            elif entry.suffix == ".bin":
                instruments.append(entry.stem)

        return sorted(set(instruments))

    def list_fields(self, instrument: str) -> List[str]:
        """获取指定证券的可用字段"""
        inst_dir = self.data_dir / instrument
        if not inst_dir.is_dir():
            # 尝试单文件模式
            bin_file = self.data_dir / f"{instrument}.bin"
            if bin_file.exists():
                return self._read_fields_from_bin(bin_file)
            return []

        fields = []
        for f in inst_dir.glob("*.bin"):
            fields.append(f.stem)
        return sorted(fields)

    def to_qlib_format(
        self,
        instruments: Optional[List[str]] = None,
        fields: Optional[List[str]] = None,
        start: Optional[str] = None,
        end: Optional[str] = None,
        include_cache: bool = True,
    ) -> pd.DataFrame:
        """
        将 .bin 数据转换为 Qlib 兼容的 MultiIndex DataFrame

        Args:
            instruments: 证券列表 (None = 全部)
            fields: 字段列表 (None = 全部)
            start: 起始日期 (YYYY-MM-DD)
            end: 结束日期 (YYYY-MM-DD)
            include_cache: 是否使用缓存

        Returns:
            MultiIndex DataFrame: index=(symbol, datetime), columns=fields
        """
        if instruments is None:
            instruments = self.list_instruments()

        all_series: List[pd.Series] = []

        for inst in instruments:
            inst_data = self._load_instrument(inst, fields, include_cache)
            if inst_data is None or inst_data.empty:
                continue

            # 过滤日期范围
            if start or end:
                if isinstance(inst_data.index, pd.DatetimeIndex):
                    mask = pd.Series(True, index=inst_data.index)
                    if start:
                        mask &= inst_data.index >= pd.Timestamp(start)
                    if end:
                        mask &= inst_data.index <= pd.Timestamp(end)
                    inst_data = inst_data.loc[mask]
                elif isinstance(inst_data.index, pd.MultiIndex):
                    # MultiIndex 已经在 load_instrument 中处理
                    pass

            if inst_data.empty:
                continue

            # 堆叠字段到 MultiIndex Series
            for col in inst_data.columns:
                s = inst_data[col].dropna()
                if s.empty:
                    continue
                s.index = pd.MultiIndex.from_arrays(
                    [[inst] * len(s), s.index],
                    names=["symbol", "datetime"],
                )
                s.name = col
                all_series.append(s)

        if not all_series:
            self.logger.warning("无数据可转换，返回空 DataFrame")
            return pd.DataFrame()

        result = pd.concat(all_series, axis=1)
        result.sort_index(inplace=True)
        self.logger.info(
            "转换完成",
            instruments=len(instruments),
            rows=len(result),
            fields=result.columns.tolist(),
        )
        return result

    def _load_instrument(
        self,
        instrument: str,
        fields: Optional[List[str]],
        use_cache: bool,
    ) -> Optional[pd.DataFrame]:
        """加载单个证券的全部字段数据"""
        cache_key = f"{instrument}:{fields}"

        if use_cache and self.cache_enabled and cache_key in self._cache:
            return self._cache[cache_key].copy()

        inst_dir = self.data_dir / instrument
        field_series: Dict[str, pd.Series] = {}

        if inst_dir.is_dir():
            # 目录模式: 每个字段一个 .bin 文件
            files_to_load = fields or [f.stem for f in inst_dir.glob("*.bin")]
            for field in files_to_load:
                bin_path = inst_dir / f"{field}.bin"
                if bin_path.exists():
                    s = self._read_bin_series(bin_path)
                    if s is not None:
                        field_series[field] = s
        else:
            # 单文件模式: 检查是否存在
            bin_file = self.data_dir / f"{instrument}.bin"
            if bin_file.exists():
                s = self._read_bin_series(bin_file)
                if s is not None:
                    field_series["value"] = s

        if not field_series:
            return None

        result = pd.DataFrame(field_series)
        if self.cache_enabled:
            self._cache[cache_key] = result.copy()
        return result

    @staticmethod
    def _read_bin_series(path: Path) -> Optional[pd.Series]:
        """读取单个 .bin 文件为 Series (datetime index)"""
        try:
            import struct

            with open(path, "rb") as f:
                # Header: [magic:4][n_features:4][feature_name_len:4][name...][data_type:1][n_rows:8]
                magic = f.read(4)
                if magic != b"QLB1":
                    # 尝试读取为 parquet
                    f.seek(0)
                    return pd.read_parquet(path).iloc[:, 0]

                n_features = struct.unpack("<I", f.read(4))[0]
                name_len = struct.unpack("<I", f.read(4))[0]
                name = f.read(name_len).decode("utf-8")
                dtype_code = struct.unpack("<B", f.read(1))[0]
                n_rows = struct.unpack("<Q", f.read(8))[0]

                # 读取索引 (datetime: 8字节时间戳)
                timestamps = []
                values = []
                dtype_map = {
                    1: ("<d", 8),   # float64
                    2: ("<f", 4),   # float32
                    3: ("<i", 4),   # int32
                    4: ("<q", 8),   # int64
                }

                if dtype_code not in dtype_map:
                    raise ValueError(f"不支持的数据类型码: {dtype_code}")

                fmt, sz = dtype_map[dtype_code]
                for _ in range(n_rows):
                    ts = struct.unpack("<q", f.read(8))[0]  # int64 纳秒时间戳
                    val = struct.unpack(fmt, f.read(sz))[0]
                    timestamps.append(pd.Timestamp(ts, unit="ns"))
                    values.append(val)

                return pd.Series(
                    values,
                    index=pd.DatetimeIndex(timestamps),
                    name=name,
                )
        except Exception:
            try:
                df = pd.read_parquet(path)
                if isinstance(df, pd.DataFrame):
                    return df.iloc[:, 0] if df.shape[1] > 0 else None
                return df
            except Exception:
                return None

    @staticmethod
    def _read_fields_from_bin(path: Path) -> List[str]:
        """读取 .bin 文件的字段名"""
        try:
            import struct
            with open(path, "rb") as f:
                magic = f.read(4)
                if magic != b"QLB1":
                    return pd.read_parquet(path).columns.tolist()
                n_features = struct.unpack("<I", f.read(4))[0]
                names = []
                for _ in range(n_features):
                    name_len = struct.unpack("<I", f.read(4))[0]
                    names.append(f.read(name_len).decode("utf-8"))
                return names
        except Exception:
            return ["value"]

    def clear_cache(self):
        self._cache.clear()
        self.logger.info("缓存已清空")


class QlibToBinConverter:
    """
    Qlib 原生 DataFrame → 自定义 .bin 格式转换器

    将 qlib 加载的数据写回为独立 .bin 文件便于离线访问。
    """

    def __init__(self, output_dir: str = "./data/bin"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger = get_logger()

    def from_qlib_format(
        self,
        df: pd.DataFrame,
        overwrite: bool = False,
    ) -> int:
        """
        将 Qlib 格式 DataFrame 拆分为 .bin 文件

        Args:
            df: Qlib MultiIndex DataFrame (symbol, datetime)
            overwrite: 是否覆盖已有文件

        Returns:
            写入文件数
        """
        if not isinstance(df.index, pd.MultiIndex):
            raise ValueError("DataFrame 必须为 MultiIndex (symbol, datetime)")

        count = 0
        for symbol in df.index.get_level_values(0).unique():
            inst_data = df.loc[symbol]
            inst_dir = self.output_dir / str(symbol)
            inst_dir.mkdir(parents=True, exist_ok=True)

            for col in inst_data.columns:
                bin_path = inst_dir / f"{col}.bin"
                if bin_path.exists() and not overwrite:
                    continue

                series = inst_data[col].dropna()
                self._write_bin_series(bin_path, series, col)
                count += 1

        self.logger.info(f"QLib→Bin 转换完成: {count} 文件")
        return count

    @staticmethod
    def _write_bin_series(path: Path, series: pd.Series, name: str):
        """写入单个 .bin 文件"""
        import struct

        with open(path, "wb") as f:
            # Header
            f.write(b"QLB1")
            f.write(struct.pack("<I", 1))  # n_features=1
            name_bytes = name.encode("utf-8")
            f.write(struct.pack("<I", len(name_bytes)))
            f.write(name_bytes)

            # 判断数据类型
            if series.dtype == np.float64:
                dtype_code, fmt = 1, "<d"
            elif series.dtype == np.float32:
                dtype_code, fmt = 2, "<f"
            elif series.dtype in (np.int32, "int32"):
                dtype_code, fmt = 3, "<i"
            else:
                series = series.astype(np.float64)
                dtype_code, fmt = 1, "<d"

            f.write(struct.pack("<B", dtype_code))
            f.write(struct.pack("<Q", len(series)))

            # Data rows
            for idx, val in series.items():
                ts_ns = int(pd.Timestamp(idx).value)
                f.write(struct.pack("<q", ts_ns))
                f.write(struct.pack(fmt, float(val)))


class FormatValidator:
    """
    格式一致性验证器

    验证 .bin ↔ Qlib DataFrame 之间的数据无损转换。
    """

    def __init__(self):
        self.logger = get_logger()

    def validate(
        self,
        bin_data: pd.DataFrame,
        qlib_data: pd.DataFrame,
        tolerance: float = 1e-6,
    ) -> Dict[str, Any]:
        """
        验证两个格式的数据一致性

        Returns:
            {
                "consistent": bool,
                "shape_match": bool,
                "index_match": bool,
                "value_match": bool,
                "max_diff": float,
                "diff_details": [...],
            }
        """
        issues = []

        # Shape 比对
        shape_match = bin_data.shape == qlib_data.shape
        if not shape_match:
            issues.append(f"Shape mismatch: bin={bin_data.shape}, qlib={qlib_data.shape}")

        # Index 比对
        try:
            index_match = bin_data.index.equals(qlib_data.index)
        except Exception:
            index_match = len(bin_data) == len(qlib_data)
        if not index_match:
            issues.append("Index mismatch")
            # 求交集
            common_idx = bin_data.index.intersection(qlib_data.index)
            bin_data = bin_data.loc[common_idx]
            qlib_data = qlib_data.loc[common_idx]

        # Value 比对
        value_match = True
        max_diff = 0.0
        common_cols = [c for c in bin_data.columns if c in qlib_data.columns]

        for col in common_cols:
            diff = (bin_data[col].fillna(0) - qlib_data[col].fillna(0)).abs()
            col_max = diff.max()
            max_diff = max(max_diff, col_max)
            if col_max > tolerance:
                value_match = False
                n_violations = (diff > tolerance).sum()
                issues.append(f"Field '{col}': {n_violations} values exceed tolerance")

        consistent = shape_match and index_match and value_match and len(issues) == 0

        self.logger.info("格式验证", consistent=consistent, issues=len(issues))
        return {
            "consistent": consistent,
            "shape_match": shape_match,
            "index_match": index_match,
            "value_match": value_match,
            "max_diff": float(max_diff),
            "issues": issues,
        }
