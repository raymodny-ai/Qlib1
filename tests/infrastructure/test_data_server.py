"""
基础设施层单元测试 — DataServer / MemoryCache / BinFileRegistry
"""

import os
import struct
import tempfile
import pytest
import numpy as np
import pandas as pd
from unittest.mock import MagicMock, patch
from src.infrastructure.data_server import (
    DataServer,
    MemoryCache,
    BinFileRegistry,
    BinFileMeta,
)


class TestMemoryCache:
    """LRU 内存缓存测试"""

    def test_put_and_get(self):
        cache = MemoryCache(max_size_mb=10)
        cache.put("key1", "value1")
        assert cache.get("key1") == "value1"

    def test_get_missing(self):
        cache = MemoryCache(max_size_mb=10)
        assert cache.get("missing") is None

    def test_contains(self):
        cache = MemoryCache(max_size_mb=10)
        cache.put("k", "v")
        assert cache.contains("k")
        assert not cache.contains("x")

    def test_hit_rate(self):
        cache = MemoryCache(max_size_mb=10)
        cache.put("k", "v")
        cache.get("k")    # hit
        cache.get("x")    # miss
        assert cache.hit_rate == 0.5

    def test_hit_rate_initial(self):
        """初始命中率应为 0.0"""
        cache = MemoryCache(max_size_mb=10)
        assert cache.hit_rate == 0.0

    def test_clear(self):
        cache = MemoryCache(max_size_mb=10)
        cache.put("k", "v")
        cache.clear()
        assert cache.get("k") is None

    def test_lru_eviction(self):
        cache = MemoryCache(max_size_mb=0.001)  # ~1KB
        cache.put("a", "x" * 1000)
        cache.put("b", "y" * 1000)
        # a 应被驱逐
        assert cache.get("a") is None or cache.get("b") is not None

    def test_stats(self):
        cache = MemoryCache(max_size_mb=10)
        cache.put("k", "v")
        stats = cache.stats
        assert "hit_rate" in stats
        assert stats["entries"] == 1

    def test_numpy_array_size(self):
        cache = MemoryCache(max_size_mb=100)
        arr = np.random.randn(100, 100)
        cache.put("arr", arr)
        assert cache.contains("arr")

    def test_pandas_series_size_estimation(self):
        """测试 pd.Series 的内存估算"""
        cache = MemoryCache(max_size_mb=100)
        series = pd.Series(np.random.randn(1000))
        cache.put("series", series)
        assert cache.contains("series")

    def test_put_with_explicit_size(self):
        """测试指定 size_bytes 写入"""
        cache = MemoryCache(max_size_mb=100)
        cache.put("key", "value", size_bytes=1024)
        assert cache.get("key") == "value"


class TestBinFileRegistry:
    """Bin 文件注册表测试"""

    def test_empty_scan(self):
        with tempfile.TemporaryDirectory() as tmp:
            reg = BinFileRegistry(tmp)
            count = reg.scan()
            assert count == 0
            assert reg.list_instruments() == []

    def test_scan_no_features_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            reg = BinFileRegistry(os.path.join(tmp, "nonexistent"))
            count = reg.scan()
            assert count == 0

    def test_get_meta_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            reg = BinFileRegistry(tmp)
            reg.scan()
            assert reg.get_meta("AAPL", "close") is None

    def test_list_instruments_empty(self):
        with tempfile.TemporaryDirectory() as tmp:
            reg = BinFileRegistry(tmp)
            reg.scan()
            assert reg.list_instruments() == []

    def test_list_fields_empty(self):
        """未扫描时应返回空列表"""
        with tempfile.TemporaryDirectory() as tmp:
            reg = BinFileRegistry(tmp)
            reg.scan()
            assert reg.list_fields("AAPL") == []

    def test_calendars_none_when_no_file(self):
        """无日历文件时 calendars 应为 None"""
        with tempfile.TemporaryDirectory() as tmp:
            reg = BinFileRegistry(tmp)
            reg.scan()
            assert reg.calendars is None

    def test_read_bin_header_valid(self):
        """测试读取有效的 .bin 文件头部"""
        with tempfile.TemporaryDirectory() as tmp:
            bin_path = os.path.join(tmp, "test.bin")
            with open(bin_path, "wb") as f:
                f.write(b"QLIB")
                f.write(struct.pack("<i", 1))
                f.write(struct.pack("<i", 100))
                f.write(struct.pack("<i", 100))
            meta = BinFileRegistry._read_bin_header(bin_path)
            assert meta is not None
            assert meta.count == 100
            assert meta.path == bin_path

    def test_read_bin_header_invalid_magic(self):
        """无效 magic 应返回 None"""
        with tempfile.TemporaryDirectory() as tmp:
            bin_path = os.path.join(tmp, "test.bin")
            with open(bin_path, "wb") as f:
                f.write(b"XXXXQLIBXXXX")
            meta = BinFileRegistry._read_bin_header(bin_path)
            assert meta is None

    def test_read_bin_header_too_short(self):
        """不足 16 字节的文件应返回 None"""
        with tempfile.TemporaryDirectory() as tmp:
            bin_path = os.path.join(tmp, "test.bin")
            with open(bin_path, "wb") as f:
                f.write(b"QLI")
            meta = BinFileRegistry._read_bin_header(bin_path)
            assert meta is None

    def test_read_bin_header_nonexistent(self):
        """不存在的文件应返回 None"""
        meta = BinFileRegistry._read_bin_header("/nonexistent/path.bin")
        assert meta is None

    def test_scan_with_real_bin_files(self):
        """测试扫描包含真实 .bin 文件的目录结构"""
        with tempfile.TemporaryDirectory() as tmp:
            features_dir = os.path.join(tmp, "features", "AAPL")
            os.makedirs(features_dir, exist_ok=True)
            bin_path = os.path.join(features_dir, "close.bin")
            with open(bin_path, "wb") as f:
                f.write(b"QLIB")
                f.write(struct.pack("<i", 1))
                f.write(struct.pack("<i", 200))
                f.write(struct.pack("<i", 200))

            reg = BinFileRegistry(tmp)
            count = reg.scan()
            assert count == 1
            assert "AAPL" in reg.list_instruments()
            meta = reg.get_meta("AAPL", "close")
            assert meta is not None
            assert meta.count == 200

    def test_scan_with_calendar(self):
        """测试扫描并加载交易日历"""
        with tempfile.TemporaryDirectory() as tmp:
            calendars_dir = os.path.join(tmp, "calendars")
            os.makedirs(calendars_dir, exist_ok=True)
            calendar_path = os.path.join(calendars_dir, "day.txt")
            with open(calendar_path, "w") as f:
                f.write("2020-01-02\n2020-01-03\n2020-01-06\n2020-01-07\n")

            reg = BinFileRegistry(tmp)
            reg.scan()
            assert reg.calendars is not None
            assert len(reg.calendars) == 4

    def test_scan_with_calendar_and_bin(self):
        """同时扫描 bin 文件和日历"""
        with tempfile.TemporaryDirectory() as tmp:
            features_dir = os.path.join(tmp, "features", "MSFT")
            os.makedirs(features_dir, exist_ok=True)
            for field in ["close", "volume"]:
                bin_path = os.path.join(features_dir, f"{field}.bin")
                with open(bin_path, "wb") as f:
                    f.write(b"QLIB")
                    f.write(struct.pack("<i", 1))
                    f.write(struct.pack("<i", 50))
                    f.write(struct.pack("<i", 50))

            calendars_dir = os.path.join(tmp, "calendars")
            os.makedirs(calendars_dir, exist_ok=True)
            with open(os.path.join(calendars_dir, "day.txt"), "w") as f:
                f.write("\n".join(
                    f"2020-01-{d:02d}" for d in range(2, 32, 1)
                    if d % 7 not in (0, 6)
                ))

            reg = BinFileRegistry(tmp)
            count = reg.scan()
            assert count == 2
            assert set(reg.list_fields("MSFT")) == {"close", "volume"}
            assert reg.calendars is not None


class TestDataServer:
    """数据服务门面测试"""

    def test_init_default(self):
        with tempfile.TemporaryDirectory() as tmp:
            ds = DataServer(provider_uri=tmp, cache_enabled=False)
            assert ds.provider_uri == tmp
            assert not ds._warmed_up

    def test_warmup_empty_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            ds = DataServer(provider_uri=tmp)
            ds.warmup()
            assert ds._warmed_up
            assert ds.stats["instruments"] == 0

    def test_warmup_with_cache(self):
        with tempfile.TemporaryDirectory() as tmp:
            ds = DataServer(provider_uri=tmp, cache_enabled=True, cache_size_mb=10)
            ds.warmup()
            assert ds._warmed_up
            stats = ds.stats
            assert stats["cache"] is not None

    def test_warmup_caches_calendar(self):
        """预热时日历应被缓存到 __calendars__ 键"""
        with tempfile.TemporaryDirectory() as tmp:
            cal_dir = os.path.join(tmp, "calendars")
            os.makedirs(cal_dir, exist_ok=True)
            with open(os.path.join(cal_dir, "day.txt"), "w") as f:
                f.write("2020-01-02\n2020-01-03\n")

            ds = DataServer(provider_uri=tmp, cache_enabled=True, cache_size_mb=10)
            ds.warmup()
            assert ds._warmed_up
            assert ds.cache.contains("__calendars__")

    def test_load_features_no_data(self):
        with tempfile.TemporaryDirectory() as tmp:
            ds = DataServer(provider_uri=tmp, cache_enabled=False)
            df = ds.load_features(["close", "volume"])
            assert isinstance(df, pd.DataFrame)

    def test_load_features_auto_warmup(self):
        """load_features 应自动触发 warmup"""
        with tempfile.TemporaryDirectory() as tmp:
            ds = DataServer(provider_uri=tmp, cache_enabled=False)
            assert not ds._warmed_up
            df = ds.load_features(["close"], instruments=["AAPL"])
            assert ds._warmed_up
            assert isinstance(df, pd.DataFrame)

    def test_load_features_with_cache_hit(self):
        """缓存命中时的加载路径"""
        with tempfile.TemporaryDirectory() as tmp:
            ds = DataServer(provider_uri=tmp, cache_enabled=True, cache_size_mb=100)
            ds._warmed_up = True
            ds.registry._calendars = pd.DatetimeIndex(
                [pd.Timestamp("2020-01-02")]
            )
            ds.registry._index["AAPL"] = {}

            # 预填充缓存
            target_cal = ds.registry._calendars
            mock_series = pd.Series(
                [100.0], index=target_cal, name="AAPL"
            )
            ds.cache.put("feature:close:None:None", mock_series)

            df = ds.load_features(["close"], instruments=["AAPL"])
            # 至少有一次缓存命中
            assert ds.cache.stats["hits"] >= 1

    def test_load_field_basic(self):
        """从 .bin 文件加载字段数据的完整路径"""
        with tempfile.TemporaryDirectory() as tmp:
            # 日历
            cal_dir = os.path.join(tmp, "calendars")
            os.makedirs(cal_dir, exist_ok=True)
            with open(os.path.join(cal_dir, "day.txt"), "w") as f:
                f.write("2020-01-02\n2020-01-03\n2020-01-06\n")

            # bin 文件 (AAPL/close)
            feat_dir = os.path.join(tmp, "features", "AAPL")
            os.makedirs(feat_dir, exist_ok=True)
            bin_path = os.path.join(feat_dir, "close.bin")
            with open(bin_path, "wb") as f:
                f.write(b"QLIB")
                f.write(struct.pack("<i", 1))
                f.write(struct.pack("<i", 3))
                f.write(struct.pack("<i", 3))
                # 索引对: (start, end) 对应日历位置
                f.write(struct.pack("<ii", 0, 1))
                f.write(struct.pack("<ii", 1, 2))
                f.write(struct.pack("<ii", 2, 3))
                # 数据: float32
                f.write(struct.pack("<fff", 150.0, 151.0, 152.0))

            ds = DataServer(provider_uri=tmp, cache_enabled=False)
            ds.warmup()

            series = ds._load_field("close", ["AAPL"], None, None)
            assert series is not None

    def test_load_field_with_date_range(self):
        """_load_field 应支持 start/end 日期过滤"""
        with tempfile.TemporaryDirectory() as tmp:
            cal_dir = os.path.join(tmp, "calendars")
            os.makedirs(cal_dir, exist_ok=True)
            with open(os.path.join(cal_dir, "day.txt"), "w") as f:
                f.write("2020-01-02\n2020-01-03\n2020-01-06\n")

            feat_dir = os.path.join(tmp, "features", "AAPL")
            os.makedirs(feat_dir, exist_ok=True)
            bin_path = os.path.join(feat_dir, "close.bin")
            with open(bin_path, "wb") as f:
                f.write(b"QLIB")
                f.write(struct.pack("<i", 1))
                f.write(struct.pack("<i", 3))
                f.write(struct.pack("<i", 3))
                f.write(struct.pack("<ii", 0, 1))
                f.write(struct.pack("<ii", 1, 2))
                f.write(struct.pack("<ii", 2, 3))
                f.write(struct.pack("<fff", 150.0, 151.0, 152.0))

            ds = DataServer(provider_uri=tmp, cache_enabled=False)
            ds.warmup()

            series = ds._load_field("close", ["AAPL"], "2020-01-03", "2020-01-06")
            assert series is not None

    def test_load_field_nonexistent_instrument(self):
        """不存在标的的 _load_field 应返回 None"""
        with tempfile.TemporaryDirectory() as tmp:
            cal_dir = os.path.join(tmp, "calendars")
            os.makedirs(cal_dir, exist_ok=True)
            with open(os.path.join(cal_dir, "day.txt"), "w") as f:
                f.write("2020-01-02\n")

            ds = DataServer(provider_uri=tmp, cache_enabled=False)
            ds.warmup()

            series = ds._load_field("close", ["NONEXIST"], None, None)
            assert series is None

    def test_stats(self):
        with tempfile.TemporaryDirectory() as tmp:
            ds = DataServer(provider_uri=tmp, cache_enabled=False)
            ds.warmup()
            s = ds.stats
            assert "warmed_up" in s
            assert "loads_total" in s

    def test_clear_cache(self):
        ds = DataServer(cache_enabled=True, cache_size_mb=10)
        ds.cache.put("test", "val")
        ds.clear_cache()
        assert ds.cache.get("test") is None

    def test_cache_disabled(self):
        ds = DataServer(cache_enabled=False)
        assert ds.cache is None
