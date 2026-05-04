"""
网络与实用工具集
"""
import ast
import datetime
import hashlib
import json
import math
import operator
import platform
import sys
import uuid
from typing import Any

from langchain_core.tools import tool


# ================= 工具内部辅助函数 =================
def _check_json_depth(obj: Any, current_depth: int = 0, max_depth: int = 50) -> int:
    """递归检查JSON嵌套深度"""
    if current_depth > max_depth:
        raise ValueError(f"JSON nesting exceeds maximum depth of {max_depth}")

    if isinstance(obj, dict):
        return max(
            (_check_json_depth(v, current_depth + 1, max_depth) for v in obj.values()),
            default=current_depth
        )
    elif isinstance(obj, list):
        return max(
            (_check_json_depth(item, current_depth + 1, max_depth) for item in obj),
            default=current_depth
        )
    return current_depth


# ================= 时间 =================
@tool
def get_current_time(timezone: str = "UTC") -> dict:
    """获取当前 UTC 时间"""
    now = datetime.datetime.utcnow()
    return {
        "datetime": now.isoformat(),
        "date": now.strftime("%Y-%m-%d"),
        "time": now.strftime("%H:%M:%S"),
        "timestamp": int(now.timestamp()),
        "timezone": "UTC",
    }


# ================= UUID =================
@tool
def generate_uuid(version: int = 4) -> dict:
    """生成 UUID"""
    if version == 1:
        return {"uuid": str(uuid.uuid1()), "version": 1}
    return {"uuid": str(uuid.uuid4()), "version": 4}


# ================= 哈希 =================
@tool
def hash_text(text: str, algorithm: str = "sha256") -> dict:
    """计算文本哈希"""
    # 限制输入大小，防止内存耗尽攻击
    MAX_TEXT_BYTES = 10 * 1024 * 1024  # 10MB
    MAX_TEXT_LENGTH = 1_000_000  # 100万字符

    if len(text) > MAX_TEXT_LENGTH:
        return {"error": f"Text too long: {len(text)} characters (max {MAX_TEXT_LENGTH})"}

    try:
        text_bytes = text.encode("utf-8")
        if len(text_bytes) > MAX_TEXT_BYTES:
            return {"error": f"Text too large: {len(text_bytes)} bytes (max {MAX_TEXT_BYTES})"}
    except Exception:
        return {"error": "Text encoding failed"}

    supported = {"md5", "sha1", "sha256", "sha512"}
    if algorithm not in supported:
        return {"error": f"Unsupported algorithm: {algorithm}"}

    h = hashlib.new(algorithm)
    h.update(text_bytes)
    return {"hash": h.hexdigest(), "algorithm": algorithm}


# ================= JSON =================
@tool
def json_parse(json_string: str) -> dict:
    """解析 JSON"""
    MAX_SIZE = 1 * 1024 * 1024  # 1MB

    if not json_string or not json_string.strip():
        return {"error": "Empty JSON string"}

    if len(json_string) > MAX_SIZE:
        return {"error": f"JSON too large: {len(json_string)} bytes (max {MAX_SIZE})"}

    try:
        data = json.loads(json_string)

        # 检查嵌套深度
        try:
            depth = _check_json_depth(data)
            if depth > 50:
                return {"error": f"JSON too deeply nested: depth {depth}"}
        except ValueError as e:
            return {"error": str(e)}

        return {"result": data}
    except json.JSONDecodeError as e:
        return {"error": str(e)}
    except RecursionError:
        return {"error": "JSON parsing exceeded recursion limit"}


@tool
def json_format(data: str, indent: int = 2) -> dict:
    """格式化 JSON"""
    MAX_SIZE = 1 * 1024 * 1024  # 1MB
    MAX_INDENT = 8

    if not data or not data.strip():
        return {"error": "Empty data string"}

    if len(data) > MAX_SIZE:
        return {"error": f"Data too large: {len(data)} bytes (max {MAX_SIZE})"}

    if not isinstance(indent, int) or indent < 0 or indent > MAX_INDENT:
        indent = 2

    try:
        obj = json.loads(data)
        return {"result": json.dumps(obj, ensure_ascii=False, indent=indent)}
    except json.JSONDecodeError as e:
        return {"error": str(e)}
    except RecursionError:
        return {"error": "JSON parsing exceeded recursion limit"}


# ================= 系统信息 =================
@tool
def system_info() -> dict:
    """获取系统信息"""
    try:
        return {
            "platform": platform.system(),
            "python_version": sys.version.split()[0],  # 只返回版本号，避免信息泄露
            "architecture": platform.machine(),
        }
    except Exception:
        return {"error": "Failed to retrieve system information"}


# ================= 计算器 =================
@tool
def calculate_expression(expression: str) -> dict:
    """
    🔴 STRICT TOOL: Mathematical Calculator
    You MUST use this tool for ANY numerical computation.
    DO NOT solve math manually under any circumstances.

    Trigger this tool when:
    - The user asks for calculation (e.g., "2+2", "sqrt(16)")
    - The task involves arithmetic or numeric reasoning
    - The answer requires computing a number

    Supported:
    - Operators: +, -, *, /, //, %, **
    - Functions: abs, round, min, max, pow, sqrt, sin, cos, tan, ln, log10, exp
    - Constants: pi, e

    Rules:
    - ALWAYS call this tool instead of computing yourself
    - EVEN IF the calculation seems simple
    - NEVER return a guessed or mental calculation
    """

    # ===== 限制 =====
    MAX_EXPR_LENGTH = 200
    MAX_NODES = 200
    MAX_DEPTH = 30
    MAX_NUMBER = 1e15
    MAX_POWER = 1000
    MAX_INPUT_STRING_LENGTH = 500

    if not expression or not isinstance(expression, str):
        return {"error": "Invalid expression", "expression": str(expression)[:100]}

    expression = expression.strip()
    if len(expression) > MAX_INPUT_STRING_LENGTH:
        return {"error": "Expression too long", "expression": expression[:100]}

    # ===== 运算符 =====
    operators = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.FloorDiv: operator.floordiv,
        ast.Pow: operator.pow,
        ast.Mod: operator.mod,
        ast.USub: operator.neg,
        ast.UAdd: operator.pos,
    }

    # ===== 函数 =====
    allowed_functions = {
        "abs": abs,
        "round": round,
        "min": min,
        "max": max,
        "pow": pow,
        "sqrt": math.sqrt,
        "sin": math.sin,
        "cos": math.cos,
        "tan": math.tan,
        "ln": math.log,
        "log10": math.log10,
        "exp": math.exp,
    }

    # ===== 常量 =====
    constants = {
        "pi": math.pi,
        "e": math.e,
    }

    # ===== 禁止的AST节点类型 =====
    FORBIDDEN_NODE_TYPES = (
        ast.Attribute,
        ast.Lambda,
        ast.Import,
        ast.ImportFrom,
        ast.Global,
        ast.Nonlocal,
        ast.ListComp,
        ast.SetComp,
        ast.DictComp,
        ast.GeneratorExp,
        ast.Yield,
        ast.YieldFrom,
        ast.AsyncFunctionDef,
        ast.AsyncFor,
        ast.AsyncWith,
        ast.Await,
        ast.FunctionDef,
        ast.ClassDef,
        ast.Delete,
        ast.Assign,
        ast.AugAssign,
        ast.AnnAssign,
        ast.For,
        ast.While,
        ast.If,
        ast.With,
        ast.Raise,
        ast.Try,
        ast.Assert,
        ast.Pass,
        ast.Break,
        ast.Continue,
        ast.Return,
    )

    # ===== 安全节点检查 =====
    def _check_node_safety(node: ast.AST) -> None:
        """递归检查AST节点是否安全"""
        if isinstance(node, FORBIDDEN_NODE_TYPES):
            raise ValueError(f"Forbidden syntax: {type(node).__name__}")

        for child in ast.iter_child_nodes(node):
            _check_node_safety(child)

    # ===== 数值检查 =====
    def _check_number(x: int | float) -> int | float:
        """检查数值是否在安全范围内"""
        if not isinstance(x, (int, float)):
            raise ValueError("Only numbers allowed")

        if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
            raise ValueError("Invalid number (NaN or infinity)")

        if isinstance(x, complex):
            raise ValueError("Complex numbers not allowed")

        if abs(x) > MAX_NUMBER:
            raise ValueError("Number too large")

        return x

    # ===== 结果规范化 =====
    def _normalize(x: int | float) -> int | float:
        """规范化浮点数，避免过长小数"""
        if isinstance(x, float):
            # 检查是否为整数
            if x.is_integer() and abs(x) <= 1e15:
                return int(x)
            return round(x, 12)
        return x

    # ===== AST 评估器 =====
    def _eval(node: ast.AST, depth: int = 0) -> int | float:
        """递归评估AST节点"""
        # 在每个节点评估前重新进行安全检查（纵深防御）
        if isinstance(node, FORBIDDEN_NODE_TYPES):
            raise ValueError(f"Forbidden syntax in evaluation: {type(node).__name__}")

        if depth > MAX_DEPTH:
            raise ValueError("Expression too deeply nested")

        # 常量
        if isinstance(node, ast.Constant):
            if isinstance(node.value, (int, float)):
                return _check_number(node.value)
            elif isinstance(node.value, complex):
                raise ValueError("Complex numbers not allowed")
            else:
                raise ValueError(f"Unsupported constant type: {type(node.value).__name__}")

        # 二元运算
        elif isinstance(node, ast.BinOp):
            op_type = type(node.op)

            # 幂运算特殊处理
            if isinstance(node.op, ast.Pow):
                if isinstance(node.right, ast.Constant):
                    if isinstance(node.right.value, (int, float)):
                        if abs(node.right.value) > MAX_POWER:
                            raise ValueError(f"Exponent too large: {node.right.value}")
                    elif isinstance(node.right, ast.UnaryOp):
                        # 负指数
                        if isinstance(node.right.op, ast.USub) and isinstance(node.right.operand, ast.Constant):
                            if abs(node.right.operand.value) > MAX_POWER:
                                raise ValueError(f"Exponent too large: {-node.right.operand.value}")

            left = _eval(node.left, depth + 1)
            right = _eval(node.right, depth + 1)

            op = operators.get(op_type)
            if not op:
                raise ValueError(f"Unsupported operator: {op_type.__name__}")

            # 除零检查
            if op_type in (ast.Div, ast.FloorDiv, ast.Mod) and right == 0:
                raise ValueError("Division by zero")

            result = op(left, right)
            return _check_number(result)

        # 一元运算
        elif isinstance(node, ast.UnaryOp):
            operand = _eval(node.operand, depth + 1)
            op = operators.get(type(node.op))
            if not op:
                raise ValueError(f"Unsupported unary operator: {type(node.op).__name__}")
            result = op(operand)
            return _check_number(result)

        # 函数调用
        elif isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name):
                raise ValueError("Invalid function call")

            func_name = node.func.id

            if func_name not in allowed_functions:
                raise ValueError(f"Function '{func_name}' not allowed")

            # 检查关键字参数
            if node.keywords:
                raise ValueError("Keyword arguments not allowed")

            args = [_eval(arg, depth + 1) for arg in node.args]

            # pow函数指数限制
            if func_name == "pow" and len(args) >= 2:
                if abs(args[1]) > MAX_POWER:
                    raise ValueError(f"Exponent too large in pow: {args[1]}")

            # 三角函数角度限制（防止极度大数导致精度问题）
            if func_name in ("sin", "cos", "tan"):
                if any(abs(arg) > MAX_NUMBER for arg in args):
                    raise ValueError("Trigonometric argument too large")

            try:
                result = allowed_functions[func_name](*args)
            except (ValueError, OverflowError, ZeroDivisionError) as e:
                raise ValueError(f"Function '{func_name}' error: {str(e)}")

            return _check_number(result)

        # 变量名（常量）
        elif isinstance(node, ast.Name):
            if node.id in constants:
                return constants[node.id]
            raise ValueError(f"Unknown name: '{node.id}'")

        # 表达式包装
        elif isinstance(node, ast.Expression):
            return _eval(node.body, depth + 1)

        raise ValueError(f"Unsupported syntax: {type(node).__name__}")

    # ===== 主执行逻辑 =====
    try:
        # 解析表达式
        try:
            tree = ast.parse(expression, mode="eval")
        except SyntaxError:
            return {"error": "Invalid syntax", "expression": expression}

        # 第一层安全检查：遍历所有节点
        nodes = list(ast.walk(tree))
        if len(nodes) > MAX_NODES:
            return {"error": "Expression too complex", "expression": expression}

        _check_node_safety(tree)

        # 执行计算
        result = _eval(tree)

        return {"result": _normalize(result), "expression": expression}

    except ValueError as e:
        return {"error": str(e), "expression": expression}
    except (RecursionError, MemoryError):
        return {"error": "Expression too complex (resource limit exceeded)", "expression": expression}
    except Exception as e:
        return {"error": f"Unexpected error: {str(e)}", "expression": expression}