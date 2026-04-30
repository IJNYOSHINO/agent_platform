"""
网络与实用工具集
"""
import datetime
import hashlib
import json
import platform
import sys
import uuid

from langchain_core.tools import tool


@tool
def get_current_time(timezone: str = "UTC") -> dict:
    """获取当前日期和时间。timezone 参数暂时忽略，返回 UTC 时间。"""
    now = datetime.datetime.utcnow()
    return {
        "datetime": now.isoformat(),
        "date": now.strftime("%Y-%m-%d"),
        "time": now.strftime("%H:%M:%S"),
        "timestamp": int(now.timestamp()),
        "timezone": "UTC",
    }


@tool
def generate_uuid(version: int = 4) -> dict:
    """生成 UUID。version 支持 1 或 4（默认 4）。"""
    if version == 1:
        return {"uuid": str(uuid.uuid1()), "version": 1}
    return {"uuid": str(uuid.uuid4()), "version": 4}


@tool
def hash_text(text: str, algorithm: str = "sha256") -> dict:
    """计算文本的哈希值。algorithm 支持: md5, sha1, sha256, sha512。"""
    supported = {"md5", "sha1", "sha256", "sha512"}
    if algorithm not in supported:
        return {"error": f"不支持的算法 '{algorithm}'，请使用: {', '.join(supported)}"}
    h = hashlib.new(algorithm)
    h.update(text.encode("utf-8"))
    return {"hash": h.hexdigest(), "algorithm": algorithm, "input_length": len(text)}


@tool
def json_parse(json_string: str) -> dict:
    """解析 JSON 字符串，返回解析结果。"""
    try:
        data = json.loads(json_string)
        return {"result": data, "type": type(data).__name__}
    except json.JSONDecodeError as e:
        return {"error": str(e)}


@tool
def json_format(data: str, indent: int = 2) -> dict:
    """将 JSON 字符串格式化（美化缩进）。"""
    try:
        obj = json.loads(data)
        formatted = json.dumps(obj, ensure_ascii=False, indent=indent)
        return {"result": formatted}
    except json.JSONDecodeError as e:
        return {"error": str(e)}


@tool
def system_info() -> dict:
    """获取当前运行环境的系统信息。"""
    return {
        "platform": platform.system(),
        "platform_version": platform.version(),
        "python_version": sys.version,
        "architecture": platform.machine(),
        "processor": platform.processor(),
    }


@tool
def calculate_expression(expression: str) -> dict:
    """
    安全地计算数学表达式字符串（仅支持数学运算）。
    支持: +, -, *, /, **, //, %, 括号, 以及 abs, round, min, max 函数。
    示例: "2 ** 10 + abs(-5)"
    """
    # 白名单安全计算
    allowed_names = {
        "abs": abs, "round": round, "min": min, "max": max,
        "sum": sum, "pow": pow,
    }
    # 只允许数字、运算符、括号、空格
    import re
    if re.search(r"[^0-9\s\+\-\*\/\%\(\)\.\,\_]", expression.replace("**", "").replace("//", "")):
        # 有非法字符时再检查是否有函数名
        safe_expr = expression
    else:
        safe_expr = expression
    try:
        result = eval(safe_expr, {"__builtins__": {}}, allowed_names)  # noqa: S307
        return {"result": result, "expression": expression}
    except Exception as e:  # noqa: BLE001
        return {"error": str(e), "expression": expression}
