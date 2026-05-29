"""
Qlib ExpressionOps 适配层 (ExpressionEngine ↔ Qlib 桥接)

将 ExpressionEngine 的财务公式自动转换为 Qlib ExpressionOps 子类，
注册到 Qlib 算子注册表，启用 C++ 向量化加速 + ExpressionCache。

核心组件:
- QlibOpsAdapter: 公式 → ExpressionOps 转换器
- ExpressionOpsCodeGen: 自动代码生成器 (T5c)
- QlibOpsRegistry: Qlib 算子注册表桥接 (T5d)

工作流:
    1. 研究员编写公式: "Mean($close, 20) / Ref($close, 20) - 1"
    2. ExpressionEngine.compile() 解析为 AST
    3. QlibOpsAdapter.translate() 将 AST 映射为 ExpressionOps 子类
    4. 注册到 Qlib operator registry → 享受 C++ cvectorize 加速

架构:
    ExpressionEngine (研究侧/快速验证)
        │
        ├── evaluate() → pandas 向量化计算 (兼容模式)
        │
        └── QlibOpsAdapter (生产侧/高性能)
            │
            ├── translate() → Qlib ExpressionOps 子类
            ├── register()  → Qlib operator registry
            └── 自动启用 ExpressionCache + C++ 加速

使用示例:
    from src.analyzers.qlib_ops_adapter import QlibOpsAdapter

    adapter = QlibOpsAdapter()
    ops_cls = adapter.translate("Mean($close, 20) / Ref($close, 20) - 1")
    adapter.register("momentum_20d", ops_cls)

    # 之后可在 Qlib 中直接使用:
    # from qlib.data import D
    # df = D.features(instruments, fields=["$momentum_20d"], ...)
"""

import ast
import re
import textwrap
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Type

from src.analyzers.expression_engine import (
    ExpressionCompiler,
    _BUILTIN_FUNCTIONS,
    ASTSandboxError,
)
from src.utils.logger import get_logger


# ============================================================================
#  公式 → ExpressionOps 映射表 (T5b)
# ============================================================================

# 表达式引擎内置函数 → Qlib ExpressionOps 类名映射
_FUNC_TO_QLIB_OPS: Dict[str, str] = {
    "Ref": "Ref",
    "Mean": "Mean",
    "Std": "Std",
    "Max": "Max",
    "Min": "Min",
    "Sum": "Sum",
    "Delta": "Sub",
    "PctChange": "PctChange",
    "Rank": "Rank",
    "Log": "Log",
    "Abs": "Abs",
    "Sign": "Sign",
    "CsRank": "CSRankNorm",
    "CsMean": "CSMean",
    "CsStd": "CSStd",
    "CsMax": "CSMax",
    "CsMin": "CSMin",
    "CsMedian": "CSMedian",
    "ref": "Ref",
    "mean": "Mean",
    "std": "Std",
    "max": "Max",
    "min": "Min",
    "sum": "Sum",
    "delta": "Sub",
    "pctchange": "PctChange",
    "rank": "Rank",
    "log": "Log",
    "abs": "Abs",
    "sign": "Sign",
}

# Python 二元运算符 → Qlib ExpressionOps 类名映射
_BINOP_TO_QLIB_OPS: Dict[type, str] = {
    ast.Add: "Add",
    ast.Sub: "Sub",
    ast.Mult: "Mul",
    ast.Div: "Div",
    ast.Pow: "Pow",
    ast.Gt: "Gt",
    ast.Lt: "Lt",
    ast.GtE: "Ge",
    ast.LtE: "Le",
    ast.Eq: "Eq",
    ast.NotEq: "Ne",
}


# ============================================================================
#  Auto CodeGen: 公式 → ExpressionOps 子类 (T5c)
# ============================================================================

@dataclass
class OpsIR:
    """ExpressionOps 中间表示 (Intermediate Representation)"""
    op_name: str                                    # Qlib operator 类名
    operands: List["OpsIR"] = field(default_factory=list)  # 子操作数
    params: Dict[str, Any] = field(default_factory=dict)   # 参数 (窗口天数等)
    field_ref: Optional[str] = None                 # 字段引用 (如 "close")


class ExpressionOpsCodeGen:
    """
    T5c: 公式 → ExpressionOps 子类自动代码生成器

    将 ExpressionEngine 的 AST 转换为 Qlib ExpressionOps 算子链，
    自动生成继承自 ExpressionOps 的 Python 类定义。

    生成的 ExpressionOps 子类:
    - 可直接注册到 Qlib operator registry
    - 享受 Qlib ExpressionCache 缓存
    - C++ cvectorize 向量化执行
    """

    # 编译后的算子类缓存
    _generated_classes: Dict[str, Type] = {}

    def __init__(self):
        self.logger = get_logger()
        self._compiler = ExpressionCompiler()

    def generate(
        self,
        formula: str,
        class_name: str = "CustomFactor",
        fields: Optional[List[str]] = None,
    ) -> Type:
        """
        从公式字符串生成 ExpressionOps 子类

        Args:
            formula: 公式字符串，如 "Mean($close, 20) / Ref($close, 20) - 1"
            class_name: 生成的类名
            fields: 显式字段声明 (可选，自动从公式提取)

        Returns:
            自动生成的 ExpressionOps 子类

        Example:
            gen = ExpressionOpsCodeGen()
            Momentum20d = gen.generate("$close / Ref($close, 20) - 1", "Momentum20d")
        """
        cache_key = f"{class_name}:{formula}"
        if cache_key in self._generated_classes:
            return self._generated_classes[cache_key]

        # 编译公式 → AST
        compiled = self._compiler.compile(formula)

        # AST → IR
        ir_tree = self._ast_to_ir(compiled.ast_tree.body)

        # IR → ExpressionOps 子类代码
        ops_deps = self._extract_ops_deps(ir_tree)
        fields_list = fields or sorted(compiled.required_fields)

        class_code = self._generate_class_code(
            class_name=class_name,
            formula=formula,
            ir_tree=ir_tree,
            fields=fields_list,
            ops_deps=ops_deps,
        )

        # 执行代码生成
        cls = self._exec_class_code(class_name, class_code, ops_deps)

        self._generated_classes[cache_key] = cls
        self.logger.info(
            f"ExpressionOps 子类已生成: {class_name}",
            formula=formula[:80],
            fields=fields_list,
            ops_count=len(ops_deps),
        )

        return cls

    def _ast_to_ir(self, node: ast.AST) -> OpsIR:
        """将 AST 节点转换为 OpsIR 中间表示"""
        if isinstance(node, ast.Constant):
            return OpsIR(op_name="Constant", params={"value": node.value})

        elif isinstance(node, ast.Name):
            # 字段引用: _F_close → close
            name = node.id
            if name.startswith("_F_"):
                return OpsIR(op_name="Field", field_ref=name[3:])
            return OpsIR(op_name="Name", params={"name": name})

        elif isinstance(node, ast.BinOp):
            op_name = _BINOP_TO_QLIB_OPS.get(type(node.op), "Unknown")
            left = self._ast_to_ir(node.left)
            right = self._ast_to_ir(node.right)
            return OpsIR(op_name=op_name, operands=[left, right])

        elif isinstance(node, ast.UnaryOp):
            if isinstance(node.op, ast.USub):
                zero = OpsIR(op_name="Constant", params={"value": 0})
                operand = self._ast_to_ir(node.operand)
                return OpsIR(op_name="Sub", operands=[zero, operand])
            return self._ast_to_ir(node.operand)

        elif isinstance(node, ast.Call):
            func_name = node.func.id if isinstance(node.func, ast.Name) else "unknown"
            qlib_op = _FUNC_TO_QLIB_OPS.get(func_name, func_name)

            args = [self._ast_to_ir(arg) for arg in node.args]
            params = {}

            # 提取时序窗口参数
            if func_name.lower() in ("ref", "mean", "std", "max", "min", "sum",
                                       "delta", "pctchange", "shift"):
                if len(args) >= 2 and args[1].op_name == "Constant":
                    params["d"] = args[1].params.get("value", 1)
                    # 窗口函数: 第一个参数是字段, 第二个是窗口
                    field_arg = args[0]
                    return OpsIR(
                        op_name=qlib_op,
                        operands=[field_arg],
                        params=params,
                        field_ref=field_arg.field_ref,
                    )

            # 普通函数调用
            return OpsIR(op_name=qlib_op, operands=args, params=params)

        elif isinstance(node, ast.Compare):
            left = self._ast_to_ir(node.left)
            results = [left]
            for op_node, comp_node in zip(node.ops, node.comparators):
                right = self._ast_to_ir(comp_node)
                op_name = _BINOP_TO_QLIB_OPS.get(type(op_node), "Unknown")
                results.append(OpsIR(op_name=op_name, operands=[results[-1], right]))
            return results[-1]

        elif isinstance(node, ast.BoolOp):
            op_name = "And" if isinstance(node.op, ast.And) else "Or"
            operands = [self._ast_to_ir(v) for v in node.values]
            return OpsIR(op_name=op_name, operands=operands)

        elif isinstance(node, ast.IfExp):
            test = self._ast_to_ir(node.test)
            body = self._ast_to_ir(node.body)
            orelse = self._ast_to_ir(node.orelse)
            return OpsIR(
                op_name="If",
                operands=[test, body, orelse],
            )

        raise TypeError(f"不支持的 AST 节点: {type(node).__name__}")

    @staticmethod
    def _extract_ops_deps(ir: OpsIR) -> Set[str]:
        """从 IR 树提取所有依赖的 Qlib ops 类名"""
        deps = set()

        def walk(node: OpsIR):
            if node.op_name not in ("Constant", "Field", "Name"):
                deps.add(node.op_name)
            for child in node.operands:
                walk(child)

        walk(ir)
        return deps

    def _generate_class_code(
        self,
        class_name: str,
        formula: str,
        ir_tree: OpsIR,
        fields: List[str],
        ops_deps: Set[str],
    ) -> str:
        """生成 ExpressionOps 子类的 Python 源代码"""
        fields_str = ", ".join(f"${f}" for f in fields)

        code = textwrap.dedent(f'''
        from qlib.data.ops import (
            ElemOperator,  ExpressionOps,
            {", ".join(sorted(ops_deps))}
        )

        class {class_name}(ElemOperator):
            """
            自动生成的 ExpressionOps 子类

            公式: {formula}
            字段: {fields_str}
            依赖 Ops: {", ".join(sorted(ops_deps)) if ops_deps else "无"}
            """

            def _load_internal(self, instrument, start_index, end_index, freq):
                {self._gen_load_internal_body(ir_tree, fields, 4)}

        __all__ = ["{class_name}"]
        ''')

        return code

    def _gen_load_internal_body(
        self,
        ir: OpsIR,
        fields: List[str],
        indent: int,
    ) -> str:
        """生成 _load_internal 方法体代码"""
        indent_str = " " * indent
        lines = []

        # 字段加载
        for f in fields:
            lines.append(f'{indent_str}{f} = Ref(${f})')
            lines.append(f'{indent_str}{f}_series = {f}.load(instrument, start_index, end_index)')

        # 运算符链
        result_var = self._ir_to_code(ir, fields, indent_str)
        lines.append(f"{indent_str}return {result_var}")

        return "\n".join(lines)

    def _ir_to_code(self, ir: OpsIR, fields: List[str], indent: str) -> str:
        """将 IR 节点转换为可执行代码字符串"""
        if ir.op_name == "Field":
            fname = ir.field_ref
            if fname and fname in fields:
                return f"{fname}_series"

        if ir.op_name == "Constant":
            return repr(ir.params.get("value", 0))

        if ir.op_name in ("Field", "Name"):
            return ir.params.get("name", "unknown")

        # 处理操作符
        operands_code = [
            self._ir_to_code(op, fields, indent)
            for op in ir.operands
        ]

        # 带窗口参数的 ops
        if "d" in ir.params and ir.operands:
            return f"{ir.op_name}({operands_code[0]}, {ir.params['d']})"

        # 二元运算符
        if len(operands_code) == 2:
            return f"{ir.op_name}({operands_code[0]}, {operands_code[1]})"

        # 一元运算符
        if len(operands_code) == 1:
            return f"{ir.op_name}({operands_code[0]})"

        return f"{ir.op_name}()"

    @staticmethod
    def _exec_class_code(
        class_name: str,
        class_code: str,
        ops_deps: Set[str],
    ) -> Type:
        """执行生成的代码并返回类对象"""
        namespace: Dict[str, Any] = {}

        # 尝试导入 Qlib ops 依赖
        try:
            import qlib.data.ops as qlib_ops
            namespace["qlib"] = __import__("qlib")

            for dep in ops_deps:
                if hasattr(qlib_ops, dep):
                    namespace[dep] = getattr(qlib_ops, dep)
            namespace["ElemOperator"] = getattr(qlib_ops, "ElemOperator", object)
            namespace["ExpressionOps"] = getattr(qlib_ops, "ExpressionOps", object)
        except ImportError:
            # Qlib 不可用: 返回占位类
            class _FallbackOps:
                def load(self, *args, **kwargs):
                    raise NotImplementedError("Qlib 未安装，无法执行 ExpressionOps")
            namespace["ElemOperator"] = _FallbackOps
            namespace["ExpressionOps"] = object
            for dep in ops_deps:
                namespace[dep] = _FallbackOps

        exec(class_code, namespace)
        return namespace[class_name]


# ============================================================================
#  QlibOpsAdapter: 主门面 (T5b)
# ============================================================================

class QlibOpsAdapter:
    """
    ExpressionEngine ↔ Qlib ExpressionOps 桥接适配器

    将金融公式转换为 Qlib ExpressionOps 子类并注册到算子注册表，
    使自定义因子能够利用 Qlib 的 C++ 加速和 ExpressionCache。

    使用示例:
        adapter = QlibOpsAdapter()

        # 方式1: 翻译 + 自动注册
        adapter.register_factor("momentum_20d", "$close / Ref($close, 20) - 1")

        # 方式2: 仅翻译 (不注册)
        ops_cls = adapter.translate("Mean($volume, 5)")

        # 方式3: 在 Qlib 中直接使用自定义因子
        # D.features(instruments, fields=["$momentum_20d"], ...)
    """

    def __init__(self):
        self.logger = get_logger()
        self._codegen = ExpressionOpsCodeGen()
        self._registered: Dict[str, Type] = {}

    def translate(
        self,
        formula: str,
        class_name: str = "CustomFactor",
    ) -> Type:
        """
        T5b: 将公式翻译为 ExpressionOps 子类

        Args:
            formula: 公式字符串
            class_name: 生成的 ExpressionOps 子类名

        Returns:
            ExpressionOps 子类
        """
        return self._codegen.generate(formula, class_name)

    def register_factor(
        self,
        factor_name: str,
        formula: str,
    ) -> Type:
        """
        T5d: 翻译公式并注册到 Qlib 算子注册表

        注册后可在 Qlib D.features() 中通过 $factor_name 直接引用。

        Args:
            factor_name: 因子名 (不含 $ 前缀)
            formula: 公式字符串

        Returns:
            已注册的 ExpressionOps 子类
        """
        class_name = f"Factor_{factor_name}"
        ops_cls = self._codegen.generate(formula, class_name)

        # 注册到 Qlib 算子注册表
        self._register_to_qlib(factor_name, ops_cls)

        self._registered[factor_name] = ops_cls
        self.logger.info(
            f"因子已注册: ${factor_name}",
            formula=formula[:80],
            class_name=class_name,
        )

        return ops_cls

    def _register_to_qlib(self, factor_name: str, ops_cls: Type):
        """
        T5d: 将自定义 ExpressionOps 注册到 Qlib 算子注册表

        通过猴子补丁方式将自定义类注入 Qlib 的 operator registry，
        使其能够被 Qlib C++ cvectorize 引擎识别并享受 ExpressionCache。
        """
        try:
            import qlib.data.ops as qlib_ops

            # 尝试将类注册到 Qlib ops 模块
            setattr(qlib_ops, ops_cls.__name__, ops_cls)

            # 尝试注入到算子注册表 (版本兼容)
            for registry_attr in ("OPERATORS", "_OPERATORS", "OPERATOR_REGISTRY"):
                registry = getattr(qlib_ops, registry_attr, None)
                if isinstance(registry, dict):
                    registry[f"${factor_name}"] = ops_cls
                    self.logger.debug(
                        f"已注册到 Qlib {registry_attr}: ${factor_name}"
                    )
                    break
            else:
                # 降级: 仅设置模块级属性
                self.logger.debug(
                    f"Qlib operator registry 未找到，"
                    f"已将 {ops_cls.__name__} 设置为模块属性"
                )

        except ImportError:
            self.logger.debug(
                "Qlib 未安装，算子注册跳过 (ExpressionEngine.evaluate() 仍可用)"
            )
        except Exception as e:
            self.logger.warning(f"Qlib 算子注册失败: {e}")

    def register_batch(
        self,
        factors: Dict[str, str],
    ) -> Dict[str, Type]:
        """
        批量注册因子

        Args:
            factors: {因子名: 公式}

        Returns:
            {因子名: ExpressionOps 子类}
        """
        results = {}
        for name, formula in factors.items():
            try:
                ops_cls = self.register_factor(name, formula)
                results[name] = ops_cls
            except Exception as e:
                self.logger.error(f"因子注册失败 [{name}]: {e}")
        return results

    @property
    def registered_factors(self) -> List[str]:
        """已注册的因子名列表"""
        return list(self._registered.keys())

    def clear_registry(self):
        """清理已注册的因子"""
        self._registered.clear()
        ExpressionOpsCodeGen._generated_classes.clear()


# ============================================================================
#  便捷函数
# ============================================================================

_default_adapter: Optional[QlibOpsAdapter] = None


def get_adapter() -> QlibOpsAdapter:
    """获取全局单例 QlibOpsAdapter"""
    global _default_adapter
    if _default_adapter is None:
        _default_adapter = QlibOpsAdapter()
    return _default_adapter


def register_factor(name: str, formula: str) -> Type:
    """便捷函数: 注册自定义因子到 Qlib"""
    return get_adapter().register_factor(name, formula)


def translate_formula(formula: str, class_name: str = "CustomFactor") -> Type:
    """便捷函数: 将公式翻译为 ExpressionOps 子类"""
    return get_adapter().translate(formula, class_name)
