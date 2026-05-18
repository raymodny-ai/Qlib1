"""
金融表达式引擎 (Expression Engine)

受 Qlib Expression Engine 启发，基于抽象语法树 (AST) 的因子计算公式编译器。
量化研究员可以用极简的数学公式字符串定义衍生变量，引擎自动处理
时间窗口滚动对齐与向量化加速计算。

支持语法:
    - 基本运算: + - * / ** ( ) , >= <= > < == !=
    - 时序函数: Ref(x, d), Mean(x, d), Std(x, d), Max(x, d), Min(x, d),
                Sum(x, d), Delta(x, d), PctChange(x, d), Shift(x, d),
                Rank(x), CsRank(x), Log(x), Abs(x), Sign(x)
    - 逻辑运算: If(cond, a, b), And(x, y), Or(x, y), Not(x)
    - 截面函数: CsMean(x), CsStd(x), CsMax(x), CsMin(x), CsMedian(x)
    - 字段引用: $close, $volume, $revenue, $net_income, ...

使用示例:
    engine = ExpressionEngine()
    # 5日动量
    result = engine.evaluate("$close / Ref($close, 5) - 1", df)
    # 布林带上轨
    result = engine.evaluate("Mean($close, 20) + 2 * Std($close, 20)", df)
    # Altman Z-Score
    result = engine.evaluate(
        "1.2 * $working_capital / $total_assets + "
        "1.4 * $retained_earnings / $total_assets + "
        "3.3 * $ebit / $total_assets + "
        "0.6 * $market_cap / $total_liabilities + "
        "1.0 * $revenue / $total_assets", df
    )
"""

import ast
import operator as op
import re
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Set, Union

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

from src.utils.logger import get_logger


# ===== 运算符映射 =====

_BINARY_OPS = {
    ast.Add: op.add,
    ast.Sub: op.sub,
    ast.Mult: op.mul,
    ast.Div: op.truediv,
    ast.Pow: op.pow,
    ast.Gt: op.gt,
    ast.Lt: op.lt,
    ast.GtE: op.ge,
    ast.LtE: op.le,
    ast.Eq: op.eq,
    ast.NotEq: op.ne,
}

_UNARY_OPS = {
    ast.USub: op.neg,
    ast.UAdd: op.pos,
    ast.Not: op.not_,
}

# ===== 内置函数实现 =====

def _ref(series: pd.Series, days: int) -> pd.Series:
    """Ref(x, d): 引用 x 在 d 天前的值。d > 0 表示过去"""
    return series.shift(days)

def _mean(series: pd.Series, window: int) -> pd.Series:
    """Mean(x, d): x 在过去 d 天的均值"""
    return series.rolling(window=abs(window), min_periods=1).mean()

def _std(series: pd.Series, window: int) -> pd.Series:
    """Std(x, d): x 在过去 d 天的标准差"""
    return series.rolling(window=abs(window), min_periods=1).std()

def _max(series: pd.Series, window: int) -> pd.Series:
    """Max(x, d): x 在过去 d 天的最大值"""
    return series.rolling(window=abs(window), min_periods=1).max()

def _min(series: pd.Series, window: int) -> pd.Series:
    """Min(x, d): x 在过去 d 天的最小值"""
    return series.rolling(window=abs(window), min_periods=1).min()

def _sum(series: pd.Series, window: int) -> pd.Series:
    """Sum(x, d): x 在过去 d 天的和"""
    return series.rolling(window=abs(window), min_periods=1).sum()

def _delta(series: pd.Series, days: int) -> pd.Series:
    """Delta(x, d): x - Ref(x, d)"""
    return series - series.shift(abs(days))

def _pct_change(series: pd.Series, days: int) -> pd.Series:
    """PctChange(x, d): (x - Ref(x, d)) / Ref(x, d)"""
    shifted = series.shift(abs(days))
    return (series - shifted) / shifted.replace(0, np.nan)

def _shift(series: pd.Series, days: int) -> pd.Series:
    """Shift(x, d): x 向前移动 d 天"""
    return series.shift(days)

def _rank(series: pd.Series) -> pd.Series:
    """Rank(x): 时间序列排名"""
    return series.rank(pct=True)

def _cs_rank(series: pd.Series, group_key: Optional[pd.Series] = None) -> pd.Series:
    """CsRank(x): 横截面排名"""
    if group_key is not None:
        return series.groupby(group_key).rank(pct=True)
    return series.rank(pct=True)

def _log(series: pd.Series) -> pd.Series:
    return np.log(series.replace(0, np.nan))

def _abs(series: pd.Series) -> pd.Series:
    return series.abs()

def _sign(series: pd.Series) -> pd.Series:
    return np.sign(series)

def _cs_mean(series: pd.Series, group_key: Optional[pd.Series] = None) -> pd.Series:
    if group_key is not None:
        return series.groupby(group_key).transform("mean")
    return pd.Series(np.full(len(series), series.mean()), index=series.index)

def _cs_std(series: pd.Series, group_key: Optional[pd.Series] = None) -> pd.Series:
    if group_key is not None:
        return series.groupby(group_key).transform("std")
    return pd.Series(np.full(len(series), series.std()), index=series.index)

def _cs_max(series: pd.Series, group_key: Optional[pd.Series] = None) -> pd.Series:
    if group_key is not None:
        return series.groupby(group_key).transform("max")
    return pd.Series(np.full(len(series), series.max()), index=series.index)

def _cs_min(series: pd.Series, group_key: Optional[pd.Series] = None) -> pd.Series:
    if group_key is not None:
        return series.groupby(group_key).transform("min")
    return pd.Series(np.full(len(series), series.min()), index=series.index)

def _cs_median(series: pd.Series, group_key: Optional[pd.Series] = None) -> pd.Series:
    if group_key is not None:
        return series.groupby(group_key).transform("median")
    return pd.Series(np.full(len(series), series.median()), index=series.index)


# 内置函数注册表
_BUILTIN_FUNCTIONS: Dict[str, Callable] = {
    "Ref": _ref,
    "Mean": _mean,
    "Std": _std,
    "Max": _max,
    "Min": _min,
    "Sum": _sum,
    "Delta": _delta,
    "PctChange": _pct_change,
    "Shift": _shift,
    "Rank": _rank,
    "CsRank": _cs_rank,
    "Log": _log,
    "Abs": _abs,
    "Sign": _sign,
    "CsMean": _cs_mean,
    "CsStd": _cs_std,
    "CsMax": _cs_max,
    "CsMin": _cs_min,
    "CsMedian": _cs_median,
    # 别名（小写）
    "ref": _ref,
    "mean": _mean,
    "std": _std,
    "max": _max,
    "min": _min,
    "sum": _sum,
    "delta": _delta,
    "pctchange": _pct_change,
    "shift": _shift,
    "rank": _rank,
    "csrank": _cs_rank,
    "log": _log,
    "abs": _abs,
    "sign": _sign,
    "csmean": _cs_mean,
    "csstd": _cs_std,
    "csmax": _cs_max,
    "csmin": _cs_min,
    "csmedian": _cs_median,
}


# ===== 表达式编译器 =====

@dataclass
class CompiledExpression:
    """编译后的表达式"""
    source: str
    ast_tree: ast.AST
    required_fields: Set[str]
    required_functions: Set[str]


class ExpressionCompiler:
    """
    AST 表达式编译器

    将表达式字符串解析为 AST，提取所需字段和函数引用。
    """

    FIELD_PATTERN = re.compile(r'\$([a-zA-Z_][a-zA-Z0-9_]*)')

    def __init__(self):
        self.logger = get_logger()

    def compile(self, expression: str) -> CompiledExpression:
        """
        编译表达式字符串

        Args:
            expression: 如 "$close / Ref($close, 5) - 1"

        Returns:
            CompiledExpression
        """
        # 预处理: 替换 $field 为合法的 Python 变量名
        fields = set()
        def replace_field(match):
            field_name = match.group(1)
            fields.add(field_name)
            return f"_F_{field_name}"

        py_expr = self.FIELD_PATTERN.sub(replace_field, expression)

        # 解析 AST
        try:
            tree = ast.parse(py_expr, mode="eval")
        except SyntaxError as e:
            self.logger.error("表达式语法错误", expression=expression, error=str(e))
            raise ValueError(f"表达式语法错误: {e}") from e

        # 提取函数引用
        functions = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                functions.add(node.func.id)

        return CompiledExpression(
            source=expression,
            ast_tree=tree,
            required_fields=fields,
            required_functions=functions,
        )


# ===== 表达式求值器 =====

class ExpressionEngine:
    """
    表达式求值引擎

    将编译后的表达式在 DataFrame 上执行向量化计算。

    使用示例:
        engine = ExpressionEngine()
        result = engine.evaluate("$close / Ref($close, 5) - 1", df)
    """

    def __init__(self, functions: Optional[Dict[str, Callable]] = None):
        """
        Args:
            functions: 额外的自定义函数注册表
        """
        self.compiler = ExpressionCompiler()
        self.functions = dict(_BUILTIN_FUNCTIONS)
        if functions:
            self.functions.update(functions)
        self.logger = get_logger()

    def evaluate(
        self,
        expression: str,
        df: pd.DataFrame,
        extra_context: Optional[Dict[str, Any]] = None,
    ) -> pd.Series:
        """
        评估表达式

        Args:
            expression: 公式字符串
            df: 数据 DataFrame (columns 为字段名)
            extra_context: 额外的上下文变量

        Returns:
            计算结果 Series
        """
        compiled = self.compiler.compile(expression)

        # 构建执行上下文
        context: Dict[str, Any] = {}

        # 映射 $field → DataFrame 列
        for field in compiled.required_fields:
            if field in df.columns:
                context[f"_F_{field}"] = df[field]
            else:
                raise KeyError(f"字段 '{field}' 不在 DataFrame 中，可用字段: {list(df.columns)}")

        # 注入内置函数
        for func_name in compiled.required_functions:
            if func_name in self.functions:
                context[func_name] = self.functions[func_name]
            elif func_name.startswith("_F_"):
                pass  # 字段引用，已在上面处理
            else:
                raise NameError(f"未知函数: {func_name}")

        # 注入额外上下文
        if extra_context:
            context.update(extra_context)

        # 执行 AST
        result = self._eval_node(compiled.ast_tree.body, context)

        if isinstance(result, pd.Series):
            result.name = expression[:50]
        elif np.isscalar(result):
            result = pd.Series(result, index=df.index, name=expression[:50])

        return result

    def _eval_node(self, node: ast.AST, context: Dict[str, Any]) -> Any:
        """递归求值 AST 节点"""
        if isinstance(node, ast.Expression):
            return self._eval_node(node.body, context)

        elif isinstance(node, ast.Constant):
            return node.value

        elif isinstance(node, ast.Name):
            name = node.id
            if name in context:
                return context[name]
            raise NameError(f"未定义变量: {name}")

        elif isinstance(node, ast.Constant):
            return node.value

        elif isinstance(node, ast.BinOp):
            left = self._eval_node(node.left, context)
            right = self._eval_node(node.right, context)
            op_func = _BINARY_OPS.get(type(node.op))
            if op_func is None:
                raise TypeError(f"不支持的二元运算符: {type(node.op)}")
            return op_func(left, right)

        elif isinstance(node, ast.UnaryOp):
            operand = self._eval_node(node.operand, context)
            op_func = _UNARY_OPS.get(type(node.op))
            if op_func is None:
                raise TypeError(f"不支持的一元运算符: {type(node.op)}")
            return op_func(operand)

        elif isinstance(node, ast.Call):
            func = self._eval_node(node.func, context)
            args = [self._eval_node(arg, context) for arg in node.args]
            kwargs = {kw.arg: self._eval_node(kw.value, context) for kw in node.keywords}

            if not callable(func):
                raise TypeError(f"'{func}' 不可调用")

            return func(*args, **kwargs)

        elif isinstance(node, ast.Compare):
            left = self._eval_node(node.left, context)
            for comp_op, comp_node in zip(node.ops, node.comparators):
                right = self._eval_node(comp_node, context)
                op_func = _BINARY_OPS.get(type(comp_op))
                if op_func is None:
                    raise TypeError(f"不支持的比较运算符: {type(comp_op)}")
                result = op_func(left, right)
                left = result  # 链式比较
            return left

        elif isinstance(node, ast.BoolOp):
            op_type = type(node.op)
            values = [self._eval_node(v, context) for v in node.values]
            if op_type == ast.And:
                result = values[0]
                for v in values[1:]:
                    result = result & v
                return result
            elif op_type == ast.Or:
                result = values[0]
                for v in values[1:]:
                    result = result | v
                return result

        elif isinstance(node, ast.IfExp):
            test = self._eval_node(node.test, context)
            if isinstance(test, pd.Series):
                true_val = self._eval_node(node.body, context)
                false_val = self._eval_node(node.orelse, context)
                true_arr = true_val.values if isinstance(true_val, pd.Series) else np.full(len(test), true_val)
                false_arr = false_val.values if isinstance(false_val, pd.Series) else np.full(len(test), false_val)
                return pd.Series(np.where(test.values, true_arr, false_arr),
                                 index=test.index)
            else:
                if test:
                    return self._eval_node(node.body, context)
                else:
                    return self._eval_node(node.orelse, context)

        elif isinstance(node, ast.Attribute):
            obj = self._eval_node(node.value, context)
            return getattr(obj, node.attr)

        raise TypeError(f"不支持的 AST 节点类型: {type(node)}")

    def evaluate_batch(
        self,
        expressions: Dict[str, str],
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        批量评估多个表达式

        Args:
            expressions: {因子名: 表达式}
            df: 数据 DataFrame

        Returns:
            DataFrame (columns = 因子名)
        """
        results = {}
        for name, expr in expressions.items():
            try:
                results[name] = self.evaluate(expr, df)
            except Exception as e:
                self.logger.error("因子计算失败", factor=name, expression=expr, error=str(e))
                results[name] = pd.Series(np.nan, index=df.index)

        return pd.DataFrame(results)
