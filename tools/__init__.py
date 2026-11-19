"""阶段三工具包：注册表 + 示例工具（知识查询 / 计算器）。"""
from .registry import TOOLS, TOOLS_BY_NAME, bind, invoke_tool, get_tools

__all__ = ["TOOLS", "TOOLS_BY_NAME", "bind", "invoke_tool", "get_tools"]
