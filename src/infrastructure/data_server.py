"""
高性能数据服务层 (Data Server)

Qlib 基础设施层的核心组件，负责对 .bin 二进制特征池与 PIT 文件数据库的
直接管理，利用多级缓存机制消除磁盘读取瓶颈。

核心组件:
- DataServer: 统一数据访问门面，Qlib 原生 API 优先 + BinFileRegistry 降级
- CacheManager: 多级缓存 (内存/Redis) 管理器
- BinFileRegistry: .bin 文件索引注册表 (降级路径)
- PITManager: PIT 时间线管理器

数据加载策略:
- 主路径: qlib.data.D.features() → Qlib C++ cvectorize 加速 + ExpressionCache
- 降级路径: BinFileRegistry 自定义 .bin 解析 (Qlib 未初始化/数据格式不兼容时)
- 二次缓存: MemoryCache (对 Qlib 返回结果做 TTL 缓存层)

设计原则:
- 缓存命中率目标 ≥ 80%
- 特征集组装 < 10s (预热后, PRD §4.1 性能红线)
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
    """LRU 内存缓存 (支持 TTL 过期)"""
    
    def __init__(self, max_size_mb: int = 4096, default_ttl: int = 3600):
        self.max_size = max_size_mb * 1024 * 1024  # 转为字节
        self._cache: OrderedDict[str, Any] = OrderedDict()
        self._timestamps: Dict[str, float] = {}    # key → 写入时间戳
        self._default_ttl = default_ttl             # 默认 TTL (秒)
        self._size: int = 0
        self._hits: int = 0
        self._misses: int = 0
        self._lock = threading.RLock()
    
    def get(self, key: str) -> Optional[Any]:
        """获取缓存项 (自动检查 TTL 过期)"""
        with self._lock:
            if key in self._cache:
                # TTL 过期检查
                ts = self._timestamps.get(key, 0)
                if ts > 0 and (time.time() - ts) > self._default_ttl:
                    self._cache.pop(key, None)
                    self._timestamps.pop(key, None)
                    self._misses += 1
                    return None
                self._cache.move_to_end(key)
                self._hits += 1
                return self._cache[key]
            self._misses += 1
            return None
    
    def put(self, key: str, value: Any, size_bytes: Optional[int] = None, ttl: Optional[int] = None):
        """写入缓存项 (可选 TTL 覆盖默认值)"""
        with self._lock:
            if size_bytes is None:
                size_bytes = self._estimate_size(value)
            
            # 驱逐旧条目直至空间足够
            while self._size + size_bytes > self.max_size and self._cache:
                oldest_key, oldest_value = self._cache.popitem(last=False)
                self._size -= self._estimate_size(oldest_value)
                self._timestamps.pop(oldest_key, None)
            
            self._cache[key] = value
            self._timestamps[key] = time.time()
            self._size += size_bytes
    
    def contains(self, key: str) -> bool:
        with self._lock:
            return key in self._cache
    
    def clear(self):
        with self._lock:
            self._cache.clear()
            self._timestamps.clear()
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
#  数据服务门面 (Qlib 原生 API 优先 + BinFileRegistry 降级)
# ============================================================================

class DataServer:
    """
    统一数据访问门面
    
    数据加载策略 (按优先级):
    1. Qlib 原生路径: qlib.data.D.features() → C++ cvectorize 加速 + ExpressionCache
    2. 降级路径: BinFileRegistry 自定义 .bin 解析 (Qlib 不可用时)
    3. 二次缓存: MemoryCache (TTL 管理, 对 Qlib 返回结果做缓存层)
    
    特性:
    - Qlib 运行时自动初始化 (qlib.init)
    - 表达式缓存预热 (warmup)
    - 命中率 ≥ 80%
    - 特征组装 < 10s (PRD §4.1 红线)
    """
    
    def __init__(
        self,
        provider_uri: str = "./data/qlib_data/us_data",
        cache_enabled: bool = True,
        cache_size_mb: int = 4096,
        cache_ttl: int = 3600,
        auto_init_qlib: bool = True,
    ):
        self.provider_uri = provider_uri
        self.registry = BinFileRegistry(provider_uri)
        self.cache = MemoryCache(max_size_mb=cache_size_mb, default_ttl=cache_ttl) if cache_enabled else None
        self.cache_ttl = cache_ttl
        self._warmed_up = False
        self._qlib_available = False
        self._qlib_initialized = False
        self._logger = get_logger(__name__)
        
        # 统计
        self._load_count: int = 0
        self._load_time_total: float = 0.0
        
        # T1.1: 尝试初始化 Qlib 运行时
        if auto_init_qlib:
            self._try_init_qlib()
    
    # ------------------------------------------------------------------
    #  Qlib 集成 (T1.1)
    # ------------------------------------------------------------------
    
    def _try_init_qlib(self) -> bool:
        """
        尝试初始化 Qlib 运行时。
        
        若成功: self._qlib_available = True, 后续 load_features() 走 Qlib 原生路径
        若失败: 回退到 BinFileRegistry 自定义解析 (降级路径)
        """
        try:
            import qlib
            from qlib.config import C
            
            # 检查 provider_uri 是否指向有效的 Qlib 数据目录
            provider_path = Path(self.provider_uri)
            calendars_dir = provider_path / "calendars"
            features_dir = provider_path / "features"
            
            if not provider_path.exists():
                self._logger.info(
                    f"Qlib 数据目录不存在 ({self.provider_uri})，"
                    f"使用 BinFileRegistry 降级路径"
                )
                return False
            
            # 初始化 Qlib (全局单例, 重复调用安全)
            qlib.init(
                provider_uri=self.provider_uri,
                expression_cache=None,  # 使用默认缓存策略
                dataset_cache=None,
                auto_mount=False,
            )
            
            self._qlib_available = True
            self._qlib_initialized = True
            
            # 获取 Qlib 日历和标的列表
            from qlib.data import D
            try:
                qlib_instruments = D.instruments(market="all")
                qlib_calendars = D.calendar()
                self._logger.info(
                    f"Qlib 初始化成功 | provider={self.provider_uri} | "
                    f"instruments={len(qlib_instruments)} | "
                    f"calendar_days={len(qlib_calendars)}"
                )
            except Exception:
                self._logger.info(f"Qlib 初始化成功 (provider={self.provider_uri})")
            
            return True
            
        except ImportError:
            self._logger.info("pyqlib 未安装，使用 BinFileRegistry 降级路径")
            return False
        except Exception as e:
            self._logger.warning(
                f"Qlib 初始化失败 ({e})，回退到 BinFileRegistry 降级路径"
            )
            return False
    
    # ------------------------------------------------------------------
    #  预热 (T1.5: Qlib 表达式缓存预热 / 降级 BinFileRegistry 扫描)
    # ------------------------------------------------------------------
    
    def warmup(self) -> None:
        """
        预热缓存。
        
        Qlib 路径: 触发表达式缓存 + 数据集缓存预热
        降级路径: 扫描 .bin 文件 + 加载日历
        """
        start = time.time()
        
        if self._qlib_available:
            self._warmup_qlib()
        else:
            self._warmup_fallback()
        
        self._warmed_up = True
        elapsed = time.time() - start
        
        instruments_count = len(self.registry.list_instruments()) if self.registry._index else 0
        self._logger.info(
            f"预热完成 | 耗时: {elapsed:.2f}s | "
            f"Qlib: {'✓' if self._qlib_available else '✗ (降级)'} | "
            f"标的: {instruments_count} | "
            f"缓存: {self.cache.stats['size_mb']:.1f}MB" if self.cache else ""
        )
    
    def _warmup_qlib(self):
        """Qlib 原生预热: 触发 ExpressionCache 和 DatasetCache"""
        try:
            from qlib.data import D
            from qlib.data.cache import ExpressionCache, DatasetCache
            
            # 预加载日历
            calendars = D.calendar()
            if self.cache and calendars is not None:
                self.cache.put("__calendars__", calendars, ttl=self.cache_ttl)
            
            # 触发表达式缓存: 对常用字段做一次预查询
            _ = D.features(
                instruments=D.instruments(market="all"),
                fields=["$close"],
                start_time="2024-01-01",
                end_time="2024-01-31",
            )
            self._logger.debug("Qlib ExpressionCache 预热完成")
            
        except Exception as e:
            self._logger.warning(f"Qlib 预热失败 ({e}), 回退到 BinFileRegistry")
            self._warmup_fallback()
    
    def _warmup_fallback(self):
        """降级预热: BinFileRegistry 扫描"""
        self.registry.scan()
        
        if self.registry.calendars is not None and self.cache:
            self.cache.put("__calendars__", self.registry.calendars, ttl=self.cache_ttl)
    
    # ------------------------------------------------------------------
    #  特征加载 (T1.2: Qlib 原生路径优先)
    # ------------------------------------------------------------------
    
    def load_features(
        self,
        fields: List[str],
        instruments: Optional[List[str]] = None,
        start: Optional[str] = None,
        end: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        加载特征矩阵
        
        优先走 Qlib 原生 API (C++ 加速 + ExpressionCache)，
        Qlib 不可用时回退到 BinFileRegistry 自定义解析。
        
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
        
        t0 = time.time()
        
        # T1.2: 主路径 — Qlib 原生 API
        if self._qlib_available:
            result = self._load_features_qlib(fields, instruments, start, end)
            if result is not None and not result.empty:
                self._load_count += 1
                self._load_time_total += time.time() - t0
                return result
            # Qlib 返回空 → 回退降级路径
            self._logger.debug("Qlib 返回空数据，回退 BinFileRegistry")
        
        # 降级路径 — BinFileRegistry
        result = self._load_features_fallback(fields, instruments, start, end)
        self._load_count += 1
        self._load_time_total += time.time() - t0
        return result
    
    def _load_features_qlib(
        self,
        fields: List[str],
        instruments: Optional[List[str]],
        start: Optional[str],
        end: Optional[str],
    ) -> Optional[pd.DataFrame]:
        """通过 Qlib 原生 API 加载特征 (T1.2)"""
        cache_key = f"qlib:features:{','.join(sorted(fields))}:{start}:{end}:{len(instruments or [])}"
        
        # T1.4: 二级缓存 — 先查 MemoryCache
        if self.cache:
            cached = self.cache.get(cache_key)
            if cached is not None:
                return cached
        
        try:
            from qlib.data import D
            
            # 构建 Qlib 表达式字段 (自动加 $ 前缀)
            qlib_fields = [
                f"${f}" if not f.startswith("$") else f
                for f in fields
            ]
            
            # 处理 instruments
            if instruments is None:
                qlib_instruments = D.instruments(market="all")
            else:
                qlib_instruments = instruments
            
            # 调用 Qlib 原生接口
            df = D.features(
                instruments=qlib_instruments,
                fields=qlib_fields,
                start_time=start,
                end_time=end,
            )
            
            if df is not None and not df.empty:
                # 去掉 $ 前缀以保持接口一致
                rename_map = {f"${f}": f for f in fields if f"${f}" in df.columns}
                if rename_map:
                    df = df.rename(columns=rename_map)
                
                # T1.4: 写入二级缓存
                if self.cache:
                    self.cache.put(cache_key, df, ttl=self.cache_ttl)
                
                return df
            
        except Exception as e:
            self._logger.debug(f"Qlib load_features 失败: {e}")
        
        return None
    
    def _load_features_fallback(
        self,
        fields: List[str],
        instruments: Optional[List[str]],
        start: Optional[str],
        end: Optional[str],
    ) -> pd.DataFrame:
        """降级路径: BinFileRegistry 自定义 .bin 解析"""
        if instruments is None:
            instruments = self.registry.list_instruments()
        
        frames: Dict[str, pd.DataFrame] = {}
        
        for field in fields:
            cache_key = f"feature:{field}:{start}:{end}"
            if self.cache and self.cache.contains(cache_key):
                series = self.cache.get(cache_key)
            else:
                series = self._load_field(field, instruments, start, end)
                if self.cache and series is not None:
                    self.cache.put(cache_key, series, ttl=self.cache_ttl)
            
            if series is not None and not series.empty:
                frames[field] = series
        
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
            "qlib_available": self._qlib_available,
            "data_path": "qlib" if self._qlib_available else "bin_registry",
            "instruments": len(self.registry.list_instruments()) if self._warmed_up else 0,
            "loads_total": self._load_count,
            "avg_load_time_s": round(self._load_time_total / max(self._load_count, 1), 4),
            "cache": self.cache.stats if self.cache else None,
        }
    
    def clear_cache(self):
        if self.cache:
            self.cache.clear()


# ========================================================================
#  K8s 入口点: python -m src.infrastructure.data_server
# ========================================================================

if __name__ == "__main__":
    import sys

    print("DataServer 启动中...")
    server = DataServer()
    server.warmup()
    stats = server.stats
    print(f"证券数: {stats.get('instruments', 0)}")
    print(f"Qlib 集成: {'✓ 已启用' if stats.get('qlib_available') else '✗ 降级路径'}")
    print(f"数据路径: {stats.get('data_path', 'unknown')}")
    print(f"预热状态: {stats.get('warmed_up', False)}")
    print(f"缓存状态: {stats.get('cache', {})}")
    print("DataServer 就绪")
    sys.exit(0)
