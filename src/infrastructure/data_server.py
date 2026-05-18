"""
高性能数据服务层 (Data Server)

Qlib 基础设施层的核心组件，负责对 .bin 二进制特征池与 PIT 文件数据库的
直接管理，利用多级缓存机制消除磁盘读取瓶颈。

核心组件:
- DataServer: 统一数据访问门面，封装 bin 文件读写与 PIT 查询
- CacheManager: 多级缓存 (内存/Redis) 管理器
- BinFileRegistry: .bin 文件索引注册表
- PITManager: PIT 时间线管理器

设计原则:
- 缓存命中率目标 ≥ 80%
- 特征集组装 < 10s (预热后)
- 线程安全的缓存读写

使用示例:
    from src.infrastructure.data_server import DataServer
    
    ds = DataServer(provider_uri="./data/qlib_data/us_data")
    ds.warmup()  # 预热缓存
    df = ds.load_features(["close", "volume"], start="2020-01-01", end="2020-12-31")
"""

import os
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np
import pandas as pd

from src.utils.logger import get_logger


# ============================================================================
#  缓存管理器
# ============================================================================

class MemoryCache:
    """LRU 内存缓存"""
    
    def __init__(self, max_size_mb: int = 4096):
        self.max_size = max_size_mb * 1024 * 1024  # 转为字节
        self._cache: OrderedDict[str, Any] = OrderedDict()
        self._size: int = 0
        self._hits: int = 0
        self._misses: int = 0
        self._lock = threading.RLock()
    
    def get(self, key: str) -> Optional[Any]:
        """获取缓存项"""
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                self._hits += 1
                return self._cache[key]
            self._misses += 1
            return None
    
    def put(self, key: str, value: Any, size_bytes: Optional[int] = None):
        """写入缓存项"""
        with self._lock:
            if size_bytes is None:
                size_bytes = self._estimate_size(value)
            
            # 驱逐旧条目直至空间足够
            while self._size + size_bytes > self.max_size and self._cache:
                oldest_key, oldest_value = self._cache.popitem(last=False)
                self._size -= self._estimate_size(oldest_value)
            
            self._cache[key] = value
            self._size += size_bytes
    
    def contains(self, key: str) -> bool:
        with self._lock:
            return key in self._cache
    
    def clear(self):
        with self._lock:
            self._cache.clear()
            self._size = 0
    
    @property
    def hit_rate(self) -> float:
        total = self._hits + self._misses
        return self._hits / total if total > 0 else 0.0
    
    @property
    def stats(self) -> Dict[str, Any]:
        return {
            "size_mb": self._size / (1024 * 1024),
            "entries": len(self._cache),
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": self.hit_rate,
        }
    
    @staticmethod
    def _estimate_size(value: Any) -> int:
        """估算对象内存占用"""
        if isinstance(value, pd.DataFrame):
            return value.memory_usage(deep=True).sum()
        elif isinstance(value, pd.Series):
            return value.memory_usage(deep=True)
        elif isinstance(value, np.ndarray):
            return value.nbytes
        return 1024  # 默认 1KB


# ============================================================================
#  Bin 文件注册表
# ============================================================================

@dataclass
class BinFileMeta:
    """.bin 文件元数据"""
    instrument: str       # 股票代码
    field: str            # 字段名 (close/open/high/low/volume/...)
    path: str             # 文件路径
    start_index: int = 0  # 起始索引偏移
    end_index: int = 0    # 结束索引偏移
    count: int = 0        # 数据点数
    index_dtype: str = "int32"
    value_dtype: str = "float32"


class BinFileRegistry:
    """.bin 文件索引注册表"""
    
    MAGIC = b"QLIB"
    HEADER_SIZE = 16  # 4(magic) + 4(version) + 4(index_cnt) + 4(data_cnt)
    
    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
        self._index: Dict[str, Dict[str, BinFileMeta]] = {}  # inst -> field -> meta
        self._calendars: Optional[pd.DatetimeIndex] = None
        self._logger = get_logger(__name__)
    
    def scan(self) -> int:
        """扫描 data_dir 下的所有 .bin 文件并建立索引"""
        count = 0
        features_dir = self.data_dir / "features"
        if features_dir.exists():
            for inst_dir in features_dir.iterdir():
                if not inst_dir.is_dir():
                    continue
                instrument = inst_dir.name
                
                for bin_file in inst_dir.glob("*.bin"):
                    field = bin_file.stem.lower()
                    meta = self._read_bin_header(bin_file)
                    if meta:
                        meta.instrument = instrument
                        meta.field = field
                        meta.path = str(bin_file)
                        
                        if instrument not in self._index:
                            self._index[instrument] = {}
                        self._index[instrument][field] = meta
                        count += 1
        
        # 加载日历
        calendar_path = self.data_dir / "calendars" / "day.txt"
        if calendar_path.exists():
            with open(calendar_path, "r") as f:
                dates = [line.strip() for line in f if line.strip()]
            self._calendars = pd.DatetimeIndex(pd.to_datetime(dates))
        
        self._logger.info(f"扫描完成: {count} 个 .bin 文件, {len(self._index)} 个标的")
        return count
    
    def get_meta(self, instrument: str, field: str) -> Optional[BinFileMeta]:
        """获取指定标的和字段的元数据"""
        inst_fields = self._index.get(instrument, {})
        return inst_fields.get(field.lower())
    
    def list_instruments(self) -> List[str]:
        return list(self._index.keys())
    
    def list_fields(self, instrument: str) -> List[str]:
        return list(self._index.get(instrument, {}).keys())
    
    @property
    def calendars(self) -> Optional[pd.DatetimeIndex]:
        return self._calendars
    
    @staticmethod
    def _read_bin_header(path: Path) -> Optional[BinFileMeta]:
        """读取 .bin 文件头部 (16字节)"""
        try:
            with open(path, "rb") as f:
                header = f.read(BinFileRegistry.HEADER_SIZE)
                if len(header) < BinFileRegistry.HEADER_SIZE:
                    return None
                
                magic = header[:4]
                if magic != BinFileRegistry.MAGIC:
                    return None
                
                version = int.from_bytes(header[4:8], "little")
                index_cnt = int.from_bytes(header[8:12], "little")
                data_cnt = int.from_bytes(header[12:16], "little")
                
                return BinFileMeta(
                    instrument="",
                    field="",
                    path=str(path),
                    count=data_cnt,
                )
        except Exception:
            return None


# ============================================================================
#  数据服务门面
# ============================================================================

class DataServer:
    """
    统一数据访问门面
    
    封装 .bin 文件读取、数据切片、PIT 查询，并提供多级缓存加速。
    """
    
    def __init__(
        self,
        provider_uri: str = "./data/qlib_data/us_data",
        cache_enabled: bool = True,
        cache_size_mb: int = 4096,
        cache_ttl: int = 3600,
    ):
        self.provider_uri = provider_uri
        self.registry = BinFileRegistry(provider_uri)
        self.cache = MemoryCache(max_size_mb=cache_size_mb) if cache_enabled else None
        self.cache_ttl = cache_ttl
        self._warmed_up = False
        self._logger = get_logger(__name__)
        
        # 统计
        self._load_count: int = 0
        self._load_time_total: float = 0.0
    
    def warmup(self) -> None:
        """预热：扫描文件 + 加载日历到缓存"""
        start = time.time()
        self.registry.scan()
        
        if self.registry.calendars is not None and self.cache:
            self.cache.put("__calendars__", self.registry.calendars)
        
        self._warmed_up = True
        elapsed = time.time() - start
        self._logger.info(f"预热完成 | 耗时: {elapsed:.2f}s | "
                         f"标的: {len(self.registry.list_instruments())} | "
                         f"缓存大小: {self.cache.stats['size_mb']:.1f}MB" if self.cache else "")
    
    def load_features(
        self,
        fields: List[str],
        instruments: Optional[List[str]] = None,
        start: Optional[str] = None,
        end: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        加载特征矩阵
        
        Args:
            fields: 特征字段列表
            instruments: 标的列表 (None = 全部)
            start: 起始日期
            end: 截止日期
            
        Returns:
            多索引 DataFrame (instrument, datetime) × fields
        """
        if not self._warmed_up:
            self.warmup()
        
        if instruments is None:
            instruments = self.registry.list_instruments()
        
        t0 = time.time()
        frames: Dict[str, pd.DataFrame] = {}
        
        for field in fields:
            cache_key = f"feature:{field}:{start}:{end}"
            if self.cache and self.cache.contains(cache_key):
                series = self.cache.get(cache_key)
            else:
                series = self._load_field(field, instruments, start, end)
                if self.cache and series is not None:
                    self.cache.put(cache_key, series)
            
            if series is not None and not series.empty:
                frames[field] = series
        
        self._load_count += 1
        self._load_time_total += time.time() - t0
        
        if not frames:
            return pd.DataFrame()
        
        # 合并为多索引 DataFrame
        result = pd.DataFrame(frames).stack(level=0).unstack(level=-1)
        return result
    
    def _load_field(
        self,
        field: str,
        instruments: List[str],
        start: Optional[str],
        end: Optional[str],
    ) -> Optional[pd.Series]:
        """从 .bin 文件加载单个字段"""
        calendars = self.registry.calendars
        if calendars is None:
            self._logger.error("日历未加载")
            return None
        
        # 日期切片
        if start:
            start_idx = calendars.get_indexer([pd.Timestamp(start)], method="bfill")[0]
        else:
            start_idx = 0
        
        if end:
            end_idx = calendars.get_indexer([pd.Timestamp(end)], method="ffill")[0] + 1
        else:
            end_idx = len(calendars)
        
        series_dict: Dict[str, pd.Series] = {}
        target_cal = calendars[start_idx:end_idx]
        
        for inst in instruments:
            meta = self.registry.get_meta(inst, field)
            if meta is None:
                continue
            
            try:
                with open(meta.path, "rb") as f:
                    f.seek(BinFileRegistry.HEADER_SIZE)
                    
                    # 读取索引数组 (int32 pairs: [start, end])
                    index_data = np.frombuffer(
                        f.read(meta.count * 8), dtype=np.int32
                    ).reshape(-1, 2)
                    
                    # 读取数据数组 (float32)
                    data = np.frombuffer(
                        f.read(meta.count * 4), dtype=np.float32
                    )
                
                # 映射到日历索引
                inst_data = np.full(len(target_cal), np.nan, dtype=np.float32)
                valid_mask = (index_data[:, 0] >= start_idx) & (index_data[:, 0] < end_idx)
                
                for idx_pair, val in zip(index_data[valid_mask], data[valid_mask]):
                    cal_pos = idx_pair[0] - start_idx
                    if 0 <= cal_pos < len(inst_data):
                        inst_data[cal_pos] = val
                
                series_dict[inst] = pd.Series(inst_data, index=target_cal, name=inst)
                
            except Exception as e:
                self._logger.debug(f"加载失败 {inst}/{field}: {e}")
                continue
        
        if not series_dict:
            return None
        
        result = pd.DataFrame(series_dict)
        return result.stack()
    
    @property
    def stats(self) -> Dict[str, Any]:
        return {
            "warmed_up": self._warmed_up,
            "instruments": len(self.registry.list_instruments()) if self._warmed_up else 0,
            "loads_total": self._load_count,
            "avg_load_time_s": self._load_time_total / max(self._load_count, 1),
            "cache": self.cache.stats if self.cache else None,
        }
    
    def clear_cache(self):
        if self.cache:
            self.cache.clear()
