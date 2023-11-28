from tools.tool_banks import *
from typing import Optional


def get_all_schemas_and_tools_by_default(azure_gpt_version: str, llm: Optional=None):
    all_schemas_and_tools_by_default = [
        code_interpreter_tools.get_PythonREPL_schema_and_tool(),
        search_tools.get_YDCSearch_schema_and_tool(azure_gpt_version=azure_gpt_version, llm=llm),
        time_tools.get_GetDatetimeTool_schema_and_tool(),
        weather_tools.get_OpenWeatherMap_schema_and_tool()
    ]
    return all_schemas_and_tools_by_default


class DefaultSchemasTools:
    def __init__(self, azure_gpt_version: str, llm: Optional=None):
        self.azure_gpt_version = azure_gpt_version
        self.llm = llm

        self.all_schemas_and_tools_by_default = get_all_schemas_and_tools_by_default(self.azure_gpt_version, self.llm)

    def schemas_and_tools(self):
        return self.all_schemas_and_tools_by_default

    def schemas(self):
        return [x[0] for x in self.all_schemas_and_tools_by_default]
    
    def tools(self):
        return [x[1] for x in self.all_schemas_and_tools_by_default]

    def tool_dictionary(self):
        return {x.name:x for x in self.tools()}
