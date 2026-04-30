"""
文本处理工具集
"""
import json
import re
from langchain_core.tools import tool


@tool
def text_length(text: str) -> dict:
    """计算文本的字符数和单词数"""
    char_count = len(text)
    word_count = len(text.split())
    line_count = len(text.splitlines())
    return {
        "char_count": char_count,
        "word_count": word_count,
        "line_count": line_count,
    }


@tool
def text_upper(text: str) -> dict:
    """将文本转换为大写"""
    return {"result": text.upper()}


@tool
def text_lower(text: str) -> dict:
    """将文本转换为小写"""
    return {"result": text.lower()}


@tool
def text_replace(text: str, old: str, new: str) -> dict:
    """在文本中将所有 old 替换为 new"""
    return {"result": text.replace(old, new), "count": text.count(old)}


@tool
def text_extract_numbers(text: str) -> dict:
    """从文本中提取所有数字"""
    numbers = re.findall(r"-?\d+\.?\d*", text)
    return {"numbers": [float(n) for n in numbers], "count": len(numbers)}


@tool
def text_split(text: str, delimiter: str = ",") -> dict:
    """按分隔符拆分文本，返回列表"""
    parts = [p.strip() for p in text.split(delimiter)]
    return {"parts": parts, "count": len(parts)}


@tool
def text_join(parts: str, separator: str = ", ") -> dict:
    """将 JSON 数组格式的字符串列表用分隔符连接"""
    try:
        items = json.loads(parts)
        if not isinstance(items, list):
            return {"error": "parts 必须是 JSON 数组格式"}
        return {"result": separator.join(str(i) for i in items)}
    except json.JSONDecodeError:
        return {"error": "无效的 JSON 数组"}


@tool
def text_summarize_stats(text: str) -> dict:
    """统计文本的基础信息：字数、段落数、最长行长度"""
    lines = text.splitlines()
    paragraphs = [p for p in text.split("\n\n") if p.strip()]
    return {
        "total_chars": len(text),
        "total_lines": len(lines),
        "total_paragraphs": len(paragraphs),
        "longest_line": max((len(l) for l in lines), default=0),
        "average_words_per_line": round(
            sum(len(l.split()) for l in lines) / max(len(lines), 1), 2
        ),
    }
