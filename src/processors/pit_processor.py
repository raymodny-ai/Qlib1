"""
Point-in-Time (PIT) 数据处理器

严格防穿越的特征数据库架构。为每项财务特征建立四维索引:
    1. instrument  — 股票代码
    2. period      — 财务周期 (如 2023-Q3)
    3. value       — 特征数值
    4. filing_date — SEC 实际发布日期 (毫秒级)

在任何给定历史回测切片下，只加载 filing_date <= 当前模拟交易日的数据版本。

核心功能:
- 财务数据 PIT 索引构建
- 防穿越时间线过滤
- 修正版本链表追踪
- 与 SEC EDGAR collector 的 PITTimelineEntry 对接
"""

import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

from src.utils.logger import get_logger

# ===== 数据结构 =====

@dataclass
class PITRecord:
    """
    单条 PIT 记录

    表示某个财务特征在某个时间点的可用值。
    同一 period 可能有多个 PITRecord（原始版本 + 修正版本），
    通过 amendment_chain 和 _next_version 形成版本链表。
    """
    instrument: str
    field: str                 # 字段名 (如 'revenue', 'net_income')
    period: str                # 财务周期 (如 '2023-Q3', '2023-FY')
    period_end_date: str       # 财务期间截止日 (YYYY-MM-DD)
    value: float
    filing_date: str           # SEC 接收日期 (YYYY-MM-DD HH:MM:SS)
    is_amended: bool = False
    amendment_chain: List[str] = field(default_factory=list)
    version_index: int = 0     # 0=原始, 1=第一次修正...


@dataclass
class PITQueryResult:
    """PIT 查询结果"""
    instrument: str
    as_of_date: str
    records: List[PITRecord] = field(default_factory=list)
    periods_available: int = 0
    records_with_amendments: int = 0


# ===== PIT 索引引擎 =====

class PITIndex:
    """
    Point-in-Time 索引引擎

    内部数据结构:
        _index: Dict[str, Dict[str, Dict[str, List[PITRecord]]]]
                instrument → field → period → [PITRecord] (按 filing_date 排序)

    查询规则:
        对给定的 as_of_date，返回每个 (field, period) 组合中
        filing_date <= as_of_date 的最新版本。
    """

    def __init__(self):
        self._index: Dict[str, Dict[str, Dict[str, List[PITRecord]]]] = {}
        self._instrument_set: Set[str] = set()
        self._field_set: Set[str] = set()
        self._period_set: Set[str] = set()
        self.logger = get_logger()

    # ===== 构建索引 =====

    def add_records(self, records: List[PITRecord]) -> int:
        """
        批量添加 PIT 记录

        Args:
            records: PITRecord 列表

        Returns:
            成功添加的记录数
        """
        count = 0
        for record in records:
            if self._add_single(record):
                count += 1
        return count

    def _add_single(self, record: PITRecord) -> bool:
        """添加单条记录到索引"""
        inst = record.instrument
        field = record.field
        period = record.period

        if inst not in self._index:
            self._index[inst] = {}
        if field not in self._index[inst]:
            self._index[inst][field] = {}
        if period not in self._index[inst][field]:
            self._index[inst][field][period] = []

        self._index[inst][field][period].append(record)
        self._instrument_set.add(inst)
        self._field_set.add(field)
        self._period_set.add(period)

        return True

    def build_from_dataframe(
        self,
        df: pd.DataFrame,
        instrument_col: str = "instrument",
        filing_date_col: str = "filing_date",
        period_col: str = "period",
        period_end_col: str = "period_end_date",
        value_fields: Optional[List[str]] = None,
    ) -> int:
        """
        从 DataFrame 构建 PIT 索引

        Args:
            df: 包含财务数据和 filing_date 的 DataFrame
            instrument_col: 股票代码列名
            filing_date_col: SEC 提交日期列名
            period_col: 财务周期列名
            period_end_col: 财务期间截止日列名
            value_fields: 要索引的数值字段列表 (None = 所有数值列)

        Returns:
            成功索引的记录数
        """
        if value_fields is None:
            # 自动识别数值列
            exclude = {instrument_col, filing_date_col, period_col, period_end_col,
                       "date", "ticker", "cik", "accession_number", "filing_type",
                       "raw_tags", "raw_xbrl_tags", "fetched_at", "statement_type"}
            value_fields = [
                c for c in df.columns
                if c not in exclude and df[c].dtype in ("float64", "float32", "int64", "int32")
            ]

        records = []
        for _, row in df.iterrows():
            instrument = str(row.get(instrument_col, ""))
            filing_date = str(row.get(filing_date_col, ""))
            period = str(row.get(period_col, ""))
            period_end = str(row.get(period_end_col, ""))

            if not instrument or not filing_date:
                continue

            for field in value_fields:
                value = row.get(field)
                if value is None or (isinstance(value, float) and np.isnan(value)):
                    continue

                records.append(PITRecord(
                    instrument=instrument,
                    field=field,
                    period=period,
                    period_end_date=period_end,
                    value=float(value),
                    filing_date=filing_date,
                ))

        return self.add_records(records)

    def sort_index(self):
        """对索引内所有记录按 filing_date 排序并建立版本索引"""
        for inst_fields in self._index.values():
            for periods in inst_fields.values():
                for record_list in periods.values():
                    record_list.sort(key=lambda r: r.filing_date)
                    for i, record in enumerate(record_list):
                        record.version_index = i
                        record.is_amended = i > 0
                        if i > 0:
                            record.amendment_chain = [
                                r.filing_date for r in record_list[:i + 1]
                            ]
                    # 建立 _next 指针
                    for i in range(len(record_list) - 1):
                        record_list[i].__dict__["_next_version"] = record_list[i + 1].filing_date

    # ===== PIT 查询 =====

    def query(
        self,
        instrument: str,
        as_of_date: str,
        fields: Optional[List[str]] = None,
        periods: Optional[List[str]] = None,
    ) -> PITQueryResult:
        """
        PIT 查询：获取指定日期可用的财务数据

        Args:
            instrument: 股票代码
            as_of_date: 模拟交易日 (YYYY-MM-DD)
            fields: 目标字段列表 (None = 全部)
            periods: 目标财务周期 (None = 全部)

        Returns:
            PITQueryResult
        """
        result = PITQueryResult(instrument=instrument, as_of_date=as_of_date)

        if instrument not in self._index:
            return result

        target_fields = fields or list(self._index[instrument].keys())

        for field in target_fields:
            if field not in self._index[instrument]:
                continue

            target_periods = periods or list(self._index[instrument][field].keys())

            for period in target_periods:
                if period not in self._index[instrument][field]:
                    continue

                # 找到 filing_date <= as_of_date 的最新记录
                records = self._index[instrument][field][period]
                valid = [r for r in records if r.filing_date[:10] <= as_of_date]

                if valid:
                    latest = valid[-1]
                    result.records.append(latest)
                    if latest.is_amended:
                        result.records_with_amendments += 1

        result.periods_available = len(set(r.period for r in result.records))
        return result

    def query_dataframe(
        self,
        instrument: str,
        as_of_date: str,
        fields: Optional[List[str]] = None,
        periods: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        PIT 查询并返回 DataFrame

        Returns:
            DataFrame: columns=[field, period, value, filing_date, is_amended, version_index]
        """
        result = self.query(instrument, as_of_date, fields, periods)
        rows = []
        for r in result.records:
            rows.append({
                "field": r.field,
                "period": r.period,
                "period_end_date": r.period_end_date,
                "value": r.value,
                "filing_date": r.filing_date,
                "is_amended": r.is_amended,
                "version_index": r.version_index,
            })
        return pd.DataFrame(rows) if rows else pd.DataFrame()

    def query_feature_matrix(
        self,
        instruments: List[str],
        as_of_date: str,
        fields: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        构建 PIT 特征矩阵

        对多个 instrument 执行 PIT 查询，返回横截面特征矩阵。

        Returns:
            DataFrame: index=instrument, columns=field (取最新 period 的值)
        """
        rows = {}
        for inst in instruments:
            result = self.query(inst, as_of_date, fields)

            if not result.records:
                continue

            row = {}
            # 对每个 field，维护当前已记录的最大 period_end_date
            # 修复: 之前用 result.records[0].period_end_date 作为比较基准
            # (该值取决于 dict 遍历顺序，完全随机)，导致静默数据污染
            latest_period_end: Dict[str, str] = {}
            for r in result.records:
                key = r.field
                current_latest = latest_period_end.get(key, "")
                if key not in row or r.period_end_date > current_latest:
                    row[key] = r.value
                    latest_period_end[key] = r.period_end_date

            if row:
                rows[inst] = row

        df = pd.DataFrame.from_dict(rows, orient="index")
        df.index.name = "instrument"
        return df

    # ===== 版本追溯 =====

    def get_amendment_history(
        self, instrument: str, field: str, period: str,
    ) -> List[PITRecord]:
        """
        获取某个 (instrument, field, period) 的完整修正历史

        Returns:
            按 filing_date 排序的 PITRecord 列表 (0=原始, 1=第一次修正...)
        """
        try:
            records = self._index[instrument][field][period]
            return sorted(records, key=lambda r: r.filing_date)
        except KeyError:
            return []

    def detect_restatements(
        self, instrument: str, as_of_date: str,
    ) -> Dict[str, List[str]]:
        """
        检测追溯调整的财务数据

        对每个 field，找出在 as_of_date 之后发布的新版本
        （这些数据在当时是不可知的）。

        Returns:
            {field_name: ["period1 (v1)", "period2 (v1)", ...]}
        """
        restatements: Dict[str, List[str]] = {}

        if instrument not in self._index:
            return restatements

        for field, periods in self._index[instrument].items():
            for period, records in periods.items():
                for record in records:
                    if record.is_amended and record.filing_date[:10] > as_of_date:
                        if field not in restatements:
                            restatements[field] = []
                        restatements[field].append(f"{period} (v{record.version_index})")

        return restatements

    # ===== 索引诊断 =====

    @property
    def summary(self) -> dict:
        """索引摘要统计"""
        total_records = 0
        amended_records = 0
        fields_per_inst: Dict[str, int] = {}

        for inst, fields in self._index.items():
            inst_count = 0
            for field, periods in fields.items():
                for records in periods.values():
                    total_records += len(records)
                    amended_records += sum(1 for r in records if r.is_amended)
                    inst_count += len(periods)
            fields_per_inst[inst] = inst_count

        return {
            "instruments": len(self._instrument_set),
            "fields": len(self._field_set),
            "periods": len(self._period_set),
            "total_records": total_records,
            "amended_records": amended_records,
            "amendment_ratio": round(amended_records / total_records, 4) if total_records else 0,
            "fields_per_instrument": fields_per_inst,
        }

    def save(self, path: str):
        """将索引序列化保存为 Parquet 文件"""
        all_records = []
        for inst, fields in self._index.items():
            for field, periods in fields.items():
                for records in periods.values():
                    for r in records:
                        all_records.append({
                            "instrument": r.instrument,
                            "field": r.field,
                            "period": r.period,
                            "period_end_date": r.period_end_date,
                            "value": r.value,
                            "filing_date": r.filing_date,
                            "is_amended": r.is_amended,
                            "version_index": r.version_index,
                        })

        df = pd.DataFrame(all_records)
        df.to_parquet(path, index=False)
        self.logger.info("PIT 索引已保存", path=path, records=len(df))

    @classmethod
    def load(cls, path: str) -> "PITIndex":
        """从 Parquet 文件加载索引"""
        df = pd.read_parquet(path)
        index = cls()

        records = []
        for _, row in df.iterrows():
            records.append(PITRecord(
                instrument=row["instrument"],
                field=row["field"],
                period=row["period"],
                period_end_date=row.get("period_end_date", ""),
                value=row["value"],
                filing_date=row["filing_date"],
                is_amended=row.get("is_amended", False),
                version_index=row.get("version_index", 0),
            ))

        index.add_records(records)
        index.sort_index()
        return index


# ===== 时间线交叉验证 =====

class PITValidator:
    """
    PIT 数据完整性验证器

    检查:
    - 时间单调性: filing_date 不能早于 period_end_date
    - 版本连续性: 版本索引必须连续 (0, 1, 2, ...)
    - 字段完整性: 核心财务字段不可缺失
    """

    CORE_FIELDS = {"revenue", "net_income", "total_assets", "total_equity"}

    def __init__(self):
        self.logger = get_logger()

    def validate(self, pit_index: PITIndex) -> Dict[str, Any]:
        """
        全面验证 PIT 索引

        Returns:
            {
                "is_valid": bool,
                "errors": [...],
                "warnings": [...],
                "stats": {...}
            }
        """
        errors = []
        warnings = []
        stats: Dict[str, Any] = {
            "instruments_checked": 0,
            "records_checked": 0,
            "temporal_violations": 0,
            "version_gaps": 0,
            "missing_core_fields": [],
        }

        for inst, fields in pit_index._index.items():
            stats["instruments_checked"] += 1

            # 检查核心字段
            missing_core = self.CORE_FIELDS - set(fields.keys())
            if missing_core:
                warnings.append(f"{inst}: 缺少核心字段 {missing_core}")
                stats["missing_core_fields"].append(inst)

            for field, periods in fields.items():
                for period, records in periods.items():
                    stats["records_checked"] += len(records)

                    # 时间单调性检查
                    period_end = records[0].period_end_date if records else ""
                    for r in records:
                        if period_end and r.filing_date[:10] < period_end:
                            stats["temporal_violations"] += 1
                            errors.append(
                                f"时间违规: {inst}/{field}/{period} "
                                f"filing_date={r.filing_date} < period_end={period_end}"
                            )

                    # 版本连续性检查
                    versions = [r.version_index for r in records]
                    if versions and versions != list(range(len(records))):
                        stats["version_gaps"] += 1
                        errors.append(
                            f"版本不连续: {inst}/{field}/{period} "
                            f"versions={versions}"
                        )

        is_valid = len(errors) == 0
        self.logger.info(
            "PIT 验证完成",
            is_valid=is_valid,
            errors=len(errors),
            warnings=len(warnings),
            instruments=stats["instruments_checked"],
            records=stats["records_checked"],
        )

        return {
            "is_valid": is_valid,
            "errors": errors,
            "warnings": warnings,
            "stats": stats,
        }
