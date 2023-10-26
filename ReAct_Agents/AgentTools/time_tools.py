import os
import load_envs
import datetime, pytz
from typing import Type
from pydantic import BaseModel, Field
from langchain.tools import BaseTool, StructuredTool

'''
OpenWeatherMap Tool
https://openweathermap.org/
'''
class GetFromDatetimeModuleArgs(BaseModel):    
    IANA_timezone: str = Field(default='Asia/Seoul', description='the region you want to know the current time in')

class GetFromDatetimeModule(StructuredTool):
    name = 'GetFromDatetimeModule'
    description = '''Return the current time based on location.\
 The input, whose type is string, is the timezone name of the current location with respect to IANA Time Zone Databasse. The default value is Asia/Seoul.\
 The return value is also string.\
 Only use this function for the current time.'''
    args_schema : Type[GetFromDatetimeModuleArgs] = GetFromDatetimeModuleArgs

    def _run(self, IANA_timezone: str)-> str:
        tz = pytz.timezone(IANA_timezone)  
        dt = datetime.datetime.now(tz)  
        return f"year-month-day: {dt.year}-{dt.month}-{dt.day}, hour-minute: {dt.hour}-{dt.minute}"

    async def _arun(self, IANA_timezone: str)-> str:
        raise NotImplementedError("Does not support async operation yet.")
