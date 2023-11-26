
import os, sys
sys.path.append('../..')
from dotenv import load_dotenv
_ = load_dotenv('../../.env')

import requests
import datetime, pytz
from typing import Type
from pydantic import BaseModel, Field


'''
OpenWeatherMap Tool
https://openweathermap.org/
'''
class OpenWeatherMap(BaseModel):    
    """Get the weather information in a given location.\
This provides only the current weather and weather forecast for up to 7 days.\
This CANNOT provide historical weather data."""
    IANA_timezone: str = Field(default='Asia/Seoul', description='IANA Timezone of which you know the current weather')
    Days_from_now: int = Field(default=0, ge=0, le=7, description='Number of days after today. 0 means today and 1 means tomorrow, and 2 means the day after tomorrow')


class OpenWeatherMapTool:
    # For backward compatibility
    name = 'OpenWeatherMap'
    description="Get the weather information in a given location.\
This provides only the current weather and weather forecast for up to 7 days.\
This CANNOT provide historical weather data."
    args_schema : Type[OpenWeatherMap] = OpenWeatherMap
    

    def run(self, IANA_timezone: str='Asia/Seoul', Days_from_now: int=0)-> str:
        WEATHER_API_KEY = os.getenv('OPENWEATHERMAP_API_KEY')
        region = IANA_timezone.split('/')[-1] if '/' in IANA_timezone else IANA_timezone  
        region = (requests.get(f"http://api.openweathermap.org/geo/1.0/direct?q={region}&limit={1}&appid={WEATHER_API_KEY}").json())[0]
        response = requests.get(f"https://api.openweathermap.org/data/3.0/onecall?lat={region['lat']}&lon={region['lon']}&exclude=minutely,hourly&appid={WEATHER_API_KEY}").json()


        timestamp = response['daily'][Days_from_now]['dt']
        dt = datetime.datetime.fromtimestamp(timestamp, pytz.timezone(response['timezone']))  
        data = response['daily'][Days_from_now]
        temperature = data['temp']['day'] - 273.15    
        humidity = response['daily'][Days_from_now]['humidity']
        wind_speed = response['daily'][Days_from_now]['wind_speed']
        weather_summary = response['daily'][Days_from_now]['summary']

        return f"timezone: {region['name']}, current time(year-month-day): {dt.year}-{dt.month}-{dt.day}, temperature: {temperature}, humidity: {humidity}, wind_speed: {wind_speed}, weather_summary: {weather_summary}."        

    async def get_weather_info(self, IANA_timezone: str='Asia/Seoul', Days_from_now: int=0)-> str:
        return self.run(IANA_timezone, Days_from_now)


    async def arun(self, IANA_timezone: str='Asia/Seoul', Days_from_now: int=0)-> str:
        result = await self.get_weather_info(IANA_timezone, Days_from_now)
        return result


def get_OpenWeatherMap_schema_and_tool():
    return OpenWeatherMap, ('OpenWeatherMap', OpenWeatherMapTool())