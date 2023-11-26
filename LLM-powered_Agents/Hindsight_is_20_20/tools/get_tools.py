from tools.tool_banks import *

def get_all_tools_by_default():
    all_tools_by_default = [
        code_interpreter_tools.get_PythonREPL_schema_and_tool(),
        search_tools.get_YDCSearch_schema_and_tool(),
        time_tools.get_GetDatetimeTool_schema_and_tool(),
        weather_tools.get_OpenWeatherMap_schema_and_tool()
    ]
    return all_tools_by_default
