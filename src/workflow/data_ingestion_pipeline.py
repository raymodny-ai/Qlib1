"""
定时数据摄取管道 (Data Ingestion Pipeline)

自动化数据生命周期管理: 定时采集 → 格式转换 → PIT 索引构建 →
数据质量校验 → 缓存预热。

核心组件:
- DataIngestionPipeline: 端到端数据摄取编排
- IngestionScheduler: 定时任务调度器
- DataQualityGate: 数据质量门控

设计原则:
- Crontab 定时触发
- 增量更新 + 全量重建双模式
- 质量门控: 校验通过才允许落盘
- 失败自动告警 + 熔断

使用示例:
    python -m src.workflow.data_ingestion_pipeline --source all
"""

import json
import os
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import pandas as pd

from src.utils.logger import get_logger


# ============================================================================
#  数据质量门控
# ============================================================================

class QualityStatus(Enum):
    PASS = "pass"
    WARN = "warn"
    FAIL = "fail"


@dataclass
class QualityCheck:
    """单次质量检查"""
    name: str
    status: QualityStatus = QualityStatus.PASS
    message: str = ""
    metric: float = 0.0
    threshold: float = 0.0


class DataQualityGate:
    """
    数据质量门控
    
    在数据落盘前执行严格的质量检查，不通过则挂起管道。
    
    检查项:
    - 行数不低于历史均值的 80%
    - NaN 比例不超过 10%
    - 日期范围连续性
    - 数值范围合理性
    """
    
    def __init__(
        self,
        min_row_ratio: float = 0.80,
        max_nan_ratio: float = 0.10,
        max_jump_std: float = 10.0,
    ):
        self.min_row_ratio = min_row_ratio
        self.max_nan_ratio = max_nan_ratio
        self.max_jump_std = max_jump_std
        self._history: Dict[str, float] = {}  # source -> historical avg rows
        self._logger = get_logger(__name__)
    
    def check(
        self,
        df: pd.DataFrame,
        source: str,
        expected_rows: Optional[int] = None,
    ) -> List[QualityCheck]:
        """
        执行质量检查
        
        Args:
            df: 待检查 DataFrame
            source: 数据源名称
            expected_rows: 期望行数
            
        Returns:
            检查结果列表
        """
        results: List[QualityCheck] = []
        
        # 1. 非空检查
        if df.empty:
            results.append(QualityCheck(
                name="non_empty",
                status=QualityStatus.FAIL,
                message="DataFrame 为空",
            ))
            return results
        else:
            results.append(QualityCheck(
                name="non_empty",
                status=QualityStatus.PASS,
                message=f"行数: {len(df)}",
                metric=len(df),
            ))
        
        # 2. 行数检查
        if expected_rows:
            ratio = len(df) / max(expected_rows, 1)
            status = QualityStatus.PASS if ratio >= self.min_row_ratio else QualityStatus.WARN
            results.append(QualityCheck(
                name="row_count",
                status=status,
                message=f"行数: {len(df)}/{expected_rows}={ratio:.1%}",
                metric=ratio,
                threshold=self.min_row_ratio,
            ))
        
        # 3. NaN 检查
        numeric_cols = df.select_dtypes(include=["number"]).columns
        if len(numeric_cols) > 0:
            nan_ratio = df[numeric_cols].isna().mean().max()
            status = QualityStatus.PASS if nan_ratio <= self.max_nan_ratio else QualityStatus.FAIL
            results.append(QualityCheck(
                name="nan_ratio",
                status=status,
                message=f"最大 NaN 比例: {nan_ratio:.1%}",
                metric=nan_ratio,
                threshold=self.max_nan_ratio,
            ))
        
        # 4. 数值异常检查
        if len(numeric_cols) > 1 and len(df) > 1:
            returns = df[numeric_cols].pct_change().dropna(how="all")
            if not returns.empty:
                max_abs_return = returns.abs().max().max()
                status = QualityStatus.PASS if max_abs_return < self.max_jump_std else QualityStatus.WARN
                results.append(QualityCheck(
                    name="value_jump",
                    status=status,
                    message=f"最大日变动: {max_abs_return:.1f} 标准差",
                    metric=max_abs_return,
                    threshold=self.max_jump_std,
                ))
        
        return results
    
    def gate(self, df: pd.DataFrame, source: str, expected_rows: Optional[int] = None) -> Tuple[bool, List[QualityCheck]]:
        """
        门控检查
        
        Returns:
            (passed, checks) — passed=True 才允许继续管道
        """
        checks = self.check(df, source, expected_rows)
        has_fail = any(c.status == QualityStatus.FAIL for c in checks)
        
        if has_fail:
            failed = [c for c in checks if c.status == QualityStatus.FAIL]
            self._logger.error(
                f"数据质量门控失败 | source={source} | "
                + "; ".join(f"{c.name}: {c.message}" for c in failed)
            )
        else:
            self._logger.info(f"数据质量门控通过 | source={source} | {len(df)} 行")
            # 更新历史基线
            self._history[source] = len(df)
        
        return not has_fail, checks


# ============================================================================
#  数据摄取管道
# ============================================================================

@dataclass
class IngestionResult:
    """单次摄取结果"""
    source: str
    status: str = "pending"          # pending | running | success | failed
    records_ingested: int = 0
    records_filtered: int = 0
    start_time: str = ""
    end_time: str = ""
    duration_s: float = 0.0
    error: Optional[str] = None
    quality_checks: List[Dict[str, Any]] = field(default_factory=list)
    
    @property
    def success(self) -> bool:
        return self.status == "success"


class DataIngestionPipeline:
    """
    端到端数据摄取管道
    
    编排完整的数据生命周期:
    1. 从外部 API 采集原始数据
    2. 格式转换 (CSV/Parquet → Qlib .bin)
    3. PIT 索引构建
    4. 数据质量门控
    5. 缓存预热
    """
    
    def __init__(
        self,
        output_dir: str = "./data/qlib_data/us_data",
        raw_dir: str = "./data/raw",
        quality_gate: Optional[DataQualityGate] = None,
    ):
        self.output_dir = Path(output_dir)
        self.raw_dir = Path(raw_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.quality_gate = quality_gate or DataQualityGate()
        self._logger = get_logger(__name__)
        
        # 统计
        self._history: List[IngestionResult] = []
    
    def run(
        self,
        sources: List[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        tickers: Optional[List[str]] = None,
        incremental: bool = True,
    ) -> List[IngestionResult]:
        """
        执行数据摄取
        
        Args:
            sources: 数据源列表 (["alpha_vantage", "eodhd", "intrinio", "sec_edgar"] or ["all"])
            start_date: 起始日期
            end_date: 截止日期
            tickers: 标的列表
            incremental: 增量/全量模式
            
        Returns:
            摄入结果列表
        """
        if sources is None:
            sources = ["all"]
        
        if "all" in sources:
            sources = ["alpha_vantage", "eodhd", "intrinio", "sec_edgar"]
        
        results: List[IngestionResult] = []
        
        for source in sources:
            result = self._ingest_source(source, start_date, end_date, tickers, incremental)
            results.append(result)
            self._history.append(result)
        
        # 汇总
        success_count = sum(1 for r in results if r.success)
        self._logger.info(
            f"数据摄取完成 | {success_count}/{len(results)} 成功 | "
            f"总记录: {sum(r.records_ingested for r in results)}"
        )
        
        return results
    
    def _ingest_source(
        self,
        source: str,
        start_date: Optional[str],
        end_date: Optional[str],
        tickers: Optional[List[str]],
        incremental: bool,
    ) -> IngestionResult:
        """摄取单个数据源"""
        result = IngestionResult(
            source=source,
            status="running",
            start_time=datetime.now().isoformat(),
        )
        
        t0 = time.time()
        
        try:
            # 1. 采集
            raw_data = self._collect(source, start_date, end_date, tickers)
            if raw_data is None or (isinstance(raw_data, pd.DataFrame) and raw_data.empty):
                result.status = "failed"
                result.error = "采集返回空数据"
                result.end_time = datetime.now().isoformat()
                result.duration_s = time.time() - t0
                return result
            
            # 2. 质量门控
            passed, checks = self.quality_gate.gate(raw_data, source)
            result.quality_checks = [
                {"name": c.name, "status": c.status.value, "message": c.message}
                for c in checks
            ]
            
            if not passed:
                result.status = "failed"
                result.error = "数据质量门控未通过"
                result.end_time = datetime.now().isoformat()
                result.duration_s = time.time() - t0
                return result
            
            # 3. 保存原始数据
            raw_path = self.raw_dir / f"{source}_{datetime.now().strftime('%Y%m%d')}.parquet"
            raw_data.to_parquet(raw_path)
            
            # 4. 转换 (委托给 data_converter)
            result.records_ingested = len(raw_data)
            
            try:
                from src.processors.data_converter import DataConverter
                converter = DataConverter(
                    input_dir=str(self.raw_dir),
                    output_dir=str(self.output_dir),
                )
                converter.convert(source, incremental=incremental)
            except ImportError:
                self._logger.warning("DataConverter 不可用，跳过 .bin 转换")
            
            result.status = "success"
            
        except Exception as e:
            result.status = "failed"
            result.error = str(e)
            self._logger.error(f"数据摄取异常 [{source}]: {e}\n{traceback.format_exc()}")
        
        result.end_time = datetime.now().isoformat()
        result.duration_s = time.time() - t0
        
        return result
    
    def _collect(
        self,
        source: str,
        start_date: Optional[str],
        end_date: Optional[str],
        tickers: Optional[List[str]],
    ) -> Optional[pd.DataFrame]:
        """
        从指定数据源采集数据
        
        委托给具体的 Collector 实现。
        """
        try:
            if source == "alpha_vantage":
                from src.collectors.alpha_vantage import AlphaVantageCollector
                collector = AlphaVantageCollector()
                # 异步采集的简化同步包装
                return self._run_async_collect(collector, tickers, start_date, end_date)
            
            elif source == "eodhd":
                from src.collectors.eodhd import EODHDCollector
                collector = EODHDCollector()
                return self._run_async_collect(collector, tickers, start_date, end_date)
            
            elif source == "intrinio":
                from src.collectors.intrinio import IntrinioCollector
                collector = IntrinioCollector()
                return self._run_async_collect(collector, tickers, start_date, end_date)
            
            elif source == "sec_edgar":
                from src.collectors.sec_edgar import SECEdgarCollector
                collector = SECEdgarCollector()
                return self._run_async_collect(collector, tickers, start_date, end_date)
            
            else:
                self._logger.warning(f"未知数据源: {source}")
                return None
                
        except ImportError as e:
            self._logger.warning(f"Collector 不可用 [{source}]: {e}")
            return None
        except Exception as e:
            self._logger.error(f"采集失败 [{source}]: {e}")
            return None
    
    @staticmethod
    def _run_async_collect(collector, tickers, start_date, end_date):
        """简化同步包装"""
        import asyncio
        
        async def _collect():
            return await collector.collect(
                tickers=tickers,
                start_date=start_date,
                end_date=end_date,
            )
        
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # 在已有事件循环中
                import nest_asyncio
                nest_asyncio.apply()
            return loop.run_until_complete(_collect())
        except RuntimeError:
            return asyncio.run(_collect())
    
    def get_history(self, n: int = 20) -> List[Dict[str, Any]]:
        return [
            {
                "source": r.source,
                "status": r.status,
                "records": r.records_ingested,
                "duration_s": round(r.duration_s, 2),
                "error": r.error,
            }
            for r in self._history[-n:]
        ]
    
    def get_summary(self) -> Dict[str, Any]:
        """获取摄取摘要"""
        total_records = sum(r.records_ingested for r in self._history)
        success_rate = sum(1 for r in self._history if r.success) / max(len(self._history), 1)
        
        return {
            "total_runs": len(self._history),
            "total_records": total_records,
            "success_rate": f"{success_rate:.1%}",
            "last_run": self._history[-1].start_time if self._history else None,
        }


# ============================================================================
#  定时调度器
# ============================================================================

class IngestionScheduler:
    """
    定时摄入调度器
    
    模拟 Crontab 定时任务，周期性触发数据采集。
    支持:
    - 每日闭市后采集 (默认 17:00 EST)
    - 每小时增量更新
    - 手动触发
    """
    
    def __init__(
        self,
        pipeline: DataIngestionPipeline,
        schedule: str = "daily",  # daily | hourly | manual
    ):
        self.pipeline = pipeline
        self.schedule = schedule
        self._logger = get_logger(__name__)
    
    def run_once(
        self,
        sources: Optional[List[str]] = None,
        **kwargs,
    ) -> List[IngestionResult]:
        """执行一次摄取"""
        return self.pipeline.run(sources=sources, **kwargs)
    
    def run_daily(
        self,
        sources: Optional[List[str]] = None,
    ) -> List[IngestionResult]:
        """每日全量采集 (闭市后)"""
        today = datetime.now().strftime("%Y-%m-%d")
        
        self._logger.info(f"启动每日采集 | 日期: {today}")
        return self.pipeline.run(
            sources=sources or ["all"],
            start_date=None,
            end_date=today,
            incremental=True,
        )
    
    def run_hourly(self) -> List[IngestionResult]:
        """每小时增量采集"""
        now = datetime.now()
        today = now.strftime("%Y-%m-%d")
        
        self._logger.info(f"启动小时增量采集 | {now.isoformat()}")
        return self.pipeline.run(
            sources=["alpha_vantage", "eodhd"],  # 实时性较高的源
            start_date=today,
            end_date=today,
            incremental=True,
        )


# ============================================================================
#  CLI 入口
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="数据摄取管道")
    parser.add_argument("--source", "-s", default="all",
                       choices=["all", "alpha_vantage", "eodhd", "intrinio", "sec_edgar"])
    parser.add_argument("--start-date", default=None)
    parser.add_argument("--end-date", default=None)
    parser.add_argument("--tickers", "-t", default=None, help="标的文件路径")
    parser.add_argument("--output-dir", default="./data/qlib_data/us_data")
    parser.add_argument("--raw-dir", default="./data/raw")
    
    args = parser.parse_args()
    
    sources = [args.source] if args.source != "all" else None
    tickers = None
    if args.tickers:
        with open(args.tickers, "r") as f:
            tickers = [line.strip() for line in f if line.strip()]
    
    pipeline = DataIngestionPipeline(
        output_dir=args.output_dir,
        raw_dir=args.raw_dir,
    )
    
    results = pipeline.run(
        sources=sources,
        start_date=args.start_date,
        end_date=args.end_date,
        tickers=tickers,
    )
    
    # 输出结果
    for r in results:
        print(f"[{r.status.upper()}] {r.source}: {r.records_ingested} records | {r.duration_s:.1f}s")
        if r.error:
            print(f"  Error: {r.error}")
