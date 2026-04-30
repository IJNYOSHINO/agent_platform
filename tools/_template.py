"""
自定义工具模板
==============
将此文件复制到 /tools 目录，修改后重启服务即可自动加载。

规则：
  1. 使用 @tool 装饰器
  2. 函数名 = 工具名（全局唯一）
  3. docstring = 工具描述（LLM 根据此描述选择工具）
  4. 所有参数必须有类型注解
  5. 返回值必须可 JSON 序列化（dict / list / str / int / float / bool）
"""
from langchain_core.tools import tool


# ── 示例 1：简单工具 ───────────────────────────────────────

@tool
def greet(name: str, language: str = "zh") -> dict:
    """根据姓名生成问候语。language: zh=中文, en=英文"""
    greetings = {
        "zh": f"你好，{name}！",
        "en": f"Hello, {name}!",
        "ja": f"こんにちは、{name}！",
    }
    return {
        "greeting": greetings.get(language, greetings["zh"]),
        "language": language,
    }


# ── 示例 2：数据处理工具 ────────────────────────────────────

@tool
def list_stats(numbers: str) -> dict:
    """
    计算数字列表的统计信息（均值、最大值、最小值、总和）。
    numbers: 逗号分隔的数字字符串，例如 "1,2,3,4,5"
    """
    try:
        nums = [float(x.strip()) for x in numbers.split(",") if x.strip()]
    except ValueError as e:
        return {"error": f"解析数字失败: {e}"}

    if not nums:
        return {"error": "列表为空"}

    return {
        "count": len(nums),
        "sum": sum(nums),
        "mean": sum(nums) / len(nums),
        "min": min(nums),
        "max": max(nums),
    }


# ── 提示：如何在工具间传递结果 ────────────────────────────

# Plan 中使用 $step_id.field 引用前置步骤输出：
#
# [
#   {"id": "s1", "tool": "list_stats", "input": {"numbers": "1,2,3"}, "depends_on": []},
#   {"id": "s2", "tool": "greet", "input": {"name": "$s1.mean"}, "depends_on": ["s1"]}
# ]
