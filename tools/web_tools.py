"""
网络与实用工具集
"""
import datetime
import hashlib
import json
import platform
import sys
import uuid
import ast
import operator
import math

from langchain_core.tools import tool


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
    supported = {"md5", "sha1", "sha256", "sha512"}
    if algorithm not in supported:
        return {"error": f"Unsupported algorithm: {algorithm}"}

    h = hashlib.new(algorithm)
    h.update(text.encode("utf-8"))
    return {"hash": h.hexdigest(), "algorithm": algorithm}


# ================= JSON =================
@tool
def json_parse(json_string: str) -> dict:
    """解析 JSON"""
    try:
        data = json.loads(json_string)
        return {"result": data}
    except json.JSONDecodeError as e:
        return {"error": str(e)}


@tool
def json_format(data: str, indent: int = 2) -> dict:
    """格式化 JSON"""
    try:
        obj = json.loads(data)
        return {"result": json.dumps(obj, ensure_ascii=False, indent=indent)}
    except json.JSONDecodeError as e:
        return {"error": str(e)}


# ================= 系统信息 =================
@tool
def system_info() -> dict:
    """获取系统信息"""
    return {
        "platform": platform.system(),
        "python_version": sys.version,
        "architecture": platform.machine(),
    }


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
    MAX_NODES = 100
    MAX_DEPTH = 20
    MAX_NUMBER = 1e10
    MAX_POWER = 100

    if not expression or len(expression) > MAX_EXPR_LENGTH:
        return {"error": "Expression too long or empty", "expression": expression}

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

    constants = {
        "pi": math.pi,
        "e": math.e,
    }

    # ===== 数值检查 =====
    def _check_number(x):
        if not isinstance(x, (int, float)):
            raise ValueError("Only numbers allowed")

        if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
            raise ValueError("Invalid number")

        if abs(x) > MAX_NUMBER:
            raise ValueError("Number too large")

        return x

    def _normalize(x):
        return round(x, 10) if isinstance(x, float) else x

    # ===== 执行 =====
    def _eval(node, depth=0):
        if depth > MAX_DEPTH:
            raise ValueError("Expression too deep")

        if isinstance(node, ast.Constant):
            return _check_number(node.value)

        elif isinstance(node, ast.BinOp):
            if isinstance(node.op, ast.Pow):
                if isinstance(node.right, ast.Constant):
                    if abs(node.right.value) > MAX_POWER:
                        raise ValueError("Exponent too large")

            left = _eval(node.left, depth + 1)
            right = _eval(node.right, depth + 1)

            op = operators.get(type(node.op))
            if not op:
                raise ValueError("Unsupported operator")

            return _check_number(op(left, right))

        elif isinstance(node, ast.UnaryOp):
            operand = _eval(node.operand, depth + 1)
            op = operators.get(type(node.op))
            return _check_number(op(operand))

        elif isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name):
                raise ValueError("Invalid function")

            name = node.func.id

            if name not in allowed_functions:
                raise ValueError(f"Function '{name}' not allowed")

            args = [_eval(arg, depth + 1) for arg in node.args]

            # 🚨 防 pow 滥用
            if name == "pow" and len(args) == 2:
                if abs(args[1]) > MAX_POWER:
                    raise ValueError("Exponent too large")

            return _check_number(allowed_functions[name](*args))

        elif isinstance(node, ast.Name):
            if node.id in constants:
                return constants[node.id]
            raise ValueError(f"Name '{node.id}' not allowed")

        elif isinstance(node, ast.Expression):
            return _eval(node.body, depth + 1)

        raise ValueError("Unsupported expression")

    try:
        tree = ast.parse(expression, mode="eval")

        nodes = list(ast.walk(tree))
        if len(nodes) > MAX_NODES:
            return {"error": "Expression too complex", "expression": expression}

        for node in nodes:
            if isinstance(node, (
                ast.Attribute,
                ast.Lambda,
                ast.Import,
                ast.ImportFrom,
                ast.Global,
                ast.Nonlocal,
            )):
                raise ValueError("Forbidden syntax")

        result = _eval(tree)

        return {"result": _normalize(result), "expression": expression}

    except Exception as e:
        return {"error": str(e), "expression": expression}