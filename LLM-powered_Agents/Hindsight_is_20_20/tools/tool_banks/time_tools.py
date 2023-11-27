import os
import datetime, pytz
from typing import Type
from pydantic import BaseModel, Field
import asyncio

class GetDatetime(BaseModel):
    """Get the current time in a given location."""    
    IANA_timezone: str = Field(default='Asia/Seoul', description="IANA Timezone of which you know the current time. Default is 'Asia/Seoul'.")


class GetDatetimeTool:
    # For backward compatibility
    name = 'GetDatetime'
    description="Get the current time in a given location."
    args_schema : Type[GetDatetime] = GetDatetime
    
    def run(self, IANA_timezone: str='Asia/Seoul')-> str:
        dt = datetime.datetime.now(pytz.timezone(IANA_timezone))  
        return f"timezone: {IANA_timezone}, current time(year-month-day-hour-minute): {dt.year}-{dt.month}-{dt.day}-{dt.hour}-{dt.minute}"

    async def get_dt(self, IANA_timezone):
        return datetime.datetime.now(pytz.timezone(IANA_timezone))

    async def arun(self, IANA_timezone: str='Asia/Seoul')-> str:
        dt = await self.get_dt(IANA_timezone)  
        return f"timezone: {IANA_timezone}, current time(year-month-day-hour-minute): {dt.year}-{dt.month}-{dt.day}-{dt.hour}-{dt.minute}"


def get_GetDatetimeTool_schema_and_tool():
    return GetDatetime, GetDatetimeTool()