from typing import List
from langchain.tools.base import BaseTool


def My_render_text_description_and_args(tools: List[BaseTool]) -> str:
    '''
    From angchain.tools.render.render_text_description_and_args
    '''
    tool_strings = []
    for i, tool in enumerate(tools):
        args_schema = str(tool.args)
        tool_strings.append(f"tool_{i+1}: {tool.name}: {tool.description}, args: {args_schema}")
    return "\n".join(tool_strings)
