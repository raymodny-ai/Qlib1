"""
DataConverter 单元测试 — .bin 格式转换器

测试覆盖:
- BinWriter 写入/读取
- BinReader 加载字段
- DataConverter OHLCV 转换
- DataConverter 复权处理
- DataConverter 基本面转换
- dump_bin 便捷函数
- ConversionReport 统计
"""

import os
import struct
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.processors.data_converter import (
    BinConfig,
    BinReader,
    BinWriter,
    ConversionReport,
    DataConverter,
    dump_bin,
)


# ===== Fixtures =====

@pytest.fixture
def sample_calendar():
    return ["2023-01-03", "2023-01-04", "2023-01-05", "2023-01-06", "2023-01-09"]


@pytest.fixture
def sample_instruments():
    return ["AAPL", "MSFT", "GOOGL"]


@pytest.fixture
def bin_writer(sample_calendar, sample_instruments):
    return BinWriter(sample_calendar, sample_instruments)


@pytest.fixture
def sample_ohlcv_df():
    """创建示例 OHLCV DataFrame（长格式）"""
    dates = pd.date_range("2023-01-03", "2023-01-09", freq="B")
    records = []
    for inst in ["AAPL", "MSFT"]:
        base = 150.0 if inst == "AAPL" else 300.0
        for i, d in enumerate(dates):
            records.append({
                "instrument": inst,
                "date": d.strftime("%Y-%m-%d"),
                "open": base + i * 2,
                "high": base + i * 2 + 1.5,
                "low": base + i * 2 - 1.0,
                "close": base + i * 2 + 0.5,
                "volume": 10000000 - i * 500000,
                "adj_factor": 1.0,
            })
    return pd.DataFrame(records)


@pytest.fixture
def sample_fundamentals_df():
    """创建示例基本面 DataFrame"""
    return pd.DataFrame([
        {"instrument": "AAPL", "date": "2023-09-30", "revenue": 383_285_000_000,
         "net_income": 96_995_000_000, "total_assets": 352_583_000_000,
         "total_equity": 62_146_000_000},
        {"instrument": "MSFT", "date": "2023-06-30", "revenue": 211_915_000_000,
         "net_income": 82_541_000_000, "total_assets": 411_976_000_000,
         "total_equity": 206_223_000_000},
    ])


# ===== BinWriter 测试 =====

class TestBinWriter:
    """BinWriter 写入测试"""

    def test_init(self, sample_calendar, sample_instruments):
        writer = BinWriter(sample_calendar, sample_instruments)
        assert writer.num_days == 5
        assert writer.num_instruments == 3
        assert writer.date_to_idx["2023-01-03"] == 0
        assert writer.inst_to_idx["AAPL"] == 0

    def test_write_field(self, bin_writer, tmp_path):
        data = {
            "AAPL": {"2023-01-03": 150.0, "2023-01-04": 152.0},
            "MSFT": {"2023-01-03": 300.0},
        }
        path = bin_writer.write_field("close", data, str(tmp_path))
        assert os.path.exists(path)
        assert path.endswith("close.bin")

    def test_write_field_generates_meta(self, bin_writer, tmp_path):
        data = {"AAPL": {"2023-01-03": 150.0}}
        bin_writer.write_field("test_field", data, str(tmp_path))
        meta_path = tmp_path / "features" / "test_field.meta"
        assert meta_path.exists()

    def test_write_field_default_nan(self, bin_writer, tmp_path):
        """未提供的日期应填充为 NaN"""
        data = {"AAPL": {"2023-01-03": 100.0}}
        bin_writer.write_field("volume", data, str(tmp_path), default_value=-999.0)

        # 读取验证
        with open(tmp_path / "features" / "volume.bin", "rb") as f:
            magic = f.read(4)
            assert magic == b"QLIB"
            version = struct.unpack("<I", f.read(4))[0]
            idx_cnt = struct.unpack("<I", f.read(4))[0]
            data_cnt = struct.unpack("<I", f.read(4))[0]
            f.read(idx_cnt * 4)  # skip index
            arr = np.frombuffer(f.read(data_cnt * 4), dtype=np.float32)

        # AAPL 第一天应为 100.0, 其余应是 -999.0
        # AAPL 占 days 0-4
        assert arr[0] == 100.0
        assert arr[1] == -999.0


# ===== BinReader 测试 =====

class TestBinReader:
    """BinReader 读取测试"""

    @pytest.fixture
    def features_dir(self, bin_writer, tmp_path):
        """创建包含 close 字段的 features 目录"""
        data = {
            "AAPL": {"2023-01-03": 150.0, "2023-01-04": 152.0, "2023-01-05": 155.0},
            "MSFT": {"2023-01-03": 300.0, "2023-01-04": 305.0},
        }
        bin_writer.write_field("close", data, str(tmp_path))
        return tmp_path / "features"

    def test_load_meta(self, features_dir, sample_calendar):
        reader = BinReader(str(features_dir), sample_calendar)
        meta = reader.load_meta("close")
        assert meta["field_name"] == "close"
        assert meta["num_instruments"] == 3
        assert meta["calendar_len"] == 5

    def test_load_field(self, features_dir, sample_calendar):
        reader = BinReader(str(features_dir), sample_calendar)
        df = reader.load_field("close")
        assert isinstance(df, pd.DataFrame)
        assert list(df.columns) == ["AAPL", "MSFT", "GOOGL"]
        assert len(df) == 5

    def test_load_field_filter_instruments(self, features_dir, sample_calendar):
        reader = BinReader(str(features_dir), sample_calendar)
        df = reader.load_field("close", instruments=["AAPL"])
        assert list(df.columns) == ["AAPL"]

    def test_load_field_date_range(self, features_dir, sample_calendar):
        reader = BinReader(str(features_dir), sample_calendar)
        df = reader.load_field("close", start_date="2023-01-04", end_date="2023-01-05")
        assert len(df) == 2  # 2023-01-04 and 2023-01-05

    def test_load_nonexistent_field_raises(self, features_dir, sample_calendar):
        reader = BinReader(str(features_dir), sample_calendar)
        with pytest.raises(FileNotFoundError):
            reader.load_field("nonexistent")

    def test_meta_cache(self, features_dir, sample_calendar):
        reader = BinReader(str(features_dir), sample_calendar)
        meta1 = reader.load_meta("close")
        meta2 = reader.load_meta("close")
        assert meta1 is meta2  # 缓存命中


# ===== DataConverter 测试 =====

class TestDataConverter:
    """DataConverter 主转换器测试"""

    @pytest.fixture
    def converter(self, tmp_path):
        # 使用临时目录作为 output_dir
        config = BinConfig(
            output_dir=str(tmp_path / "qlib_data" / "us_data"),
            calendar_path="./config/calendars/us_market.txt",
            instruments_path="./config/instruments/all_us_stocks.txt",
        )
        return DataConverter(config=config)

    def test_convert_ohlcv_basic(self, converter, sample_ohlcv_df):
        report = converter.convert_ohlcv(sample_ohlcv_df, adjust_prices=False)
        assert report.total_instruments == 2
        assert report.total_fields >= 5  # open, high, low, close, volume
        assert report.successful > 0
        assert report.failed == 0

    def test_convert_ohlcv_with_adjustment(self, converter, sample_ohlcv_df):
        report = converter.convert_ohlcv(sample_ohlcv_df, adjust_prices=True)
        # 复权因子字段应额外生成 factor.meta 文件
        features_dir = Path(converter.config.output_dir) / "features"
        factor_meta = features_dir / "factor.meta"
        assert factor_meta.exists()

    def test_convert_fundamentals(self, converter, sample_fundamentals_df):
        report = converter.convert_fundamentals(sample_fundamentals_df, category="income")
        assert report.total_instruments == 2
        assert report.total_fields >= 4
        assert report.successful > 0

    def test_convert_all(self, converter, sample_ohlcv_df, sample_fundamentals_df):
        reports = converter.convert_all(
            ohlcv_df=sample_ohlcv_df,
            income_df=sample_fundamentals_df,
        )
        assert "ohlcv" in reports
        assert "income" in reports


class TestDumpBin:
    """dump_bin 便捷函数测试"""

    def test_dump_bin_ohlcv(self, tmp_path, sample_ohlcv_df):
        output = str(tmp_path / "qlib_out")
        report = dump_bin(sample_ohlcv_df, output_dir=output, category="ohlcv", adjust_prices=False)
        assert report.successful > 0

    def test_dump_bin_fundamentals(self, tmp_path, sample_fundamentals_df):
        output = str(tmp_path / "qlib_out")
        report = dump_bin(sample_fundamentals_df, output_dir=output, category="income")
        assert report.successful > 0


class TestConversionReport:
    """ConversionReport 统计"""

    def test_report_fields(self):
        report = ConversionReport(
            total_instruments=10, successful=9, failed=1, skipped=2,
            total_fields=45, errors=["test error"], elapsed_seconds=3.5,
        )
        assert report.total_instruments == 10
        assert report.successful == 9
        assert report.failed == 1
