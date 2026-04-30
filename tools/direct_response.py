from langchain_core.tools import tool

@tool
def direct_response(text: str) -> str:
    """当无需调用其他工具时使用，直接返回结果"""
    return text