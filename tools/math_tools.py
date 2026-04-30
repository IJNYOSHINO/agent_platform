"""
数学工具集
展示如何在 /tools 目录中定义工具：
  - 使用 @tool 装饰器
  - 必须有 docstring（作为工具描述）
  - 参数需要类型注解
  - 返回值必须可 JSON 序列化
"""
from langchain_core.tools import tool


@tool
def add(a: float, b: float) -> dict:
    """计算两个数的和"""
    return {"result": a + b}


@tool
def subtract(a: float, b: float) -> dict:
    """计算两个数的差（a - b）"""
    return {"result": a - b}


@tool
def multiply(a: float, b: float) -> dict:
    """计算两个数的积"""
    return {"result": a * b}


@tool
def divide(a: float, b: float) -> dict:
    """计算两个数的商（a / b），b 不能为 0"""
    if b == 0:
        return {"error": "除数不能为零"}
    return {"result": a / b}


@tool
def power(base: float, exponent: float) -> dict:
    """计算 base 的 exponent 次方"""
    return {"result": base ** exponent}


@tool
def sqrt(x: float) -> dict:
    """计算非负数的平方根"""
    if x < 0:
        return {"error": "不能对负数开平方根"}
    return {"result": x ** 0.5}
