import os, sys
sys.path.extend(['..', '../..'])
import requests
import datetime, pytz
from typing import Type
from pydantic import BaseModel, Field
from langchain.tools import BaseTool, StructuredTool
from dotenv import load_dotenv
_ = load_dotenv('.env')


'''
OpenWeatherMap Tool
https://openweathermap.org/
'''
class GetFromOpenWeatherMapArgs(BaseModel):    
    IANA_timezone: str = Field(default='Asia/Seoul', description='the region you want to know the weather for')
    Days_from_now: int = Field(default=0, ge=0, le=7, description='Number of days after today. 0 means today and 1 means tomorrow, and 2 means the day after tomorrow')

class GetFromOpenWeatherMap(StructuredTool):
    name = 'GetFromOpenWeatherMap'
    description = '''Return the weather information based on current location.\
 Not only can you get the current weather, you can also get weather prediction up to 7 days in advance.\
 This function is two-input function.\
 Input 1: 'IANA_timezone' is the timezone name of the current location with respect to IANA Time Zone Databasse. The default value is 'Asia/Seoul' (str).\
 Input 2: 'Days_from_now', is the number of days from today. The default value is 0 (int).\
 value is string.\
 Only use this function for the current weather and weather forecast for up to 7 days.\
 This functon CANNOT provide historical weather data.'''
    args_schema : Type[GetFromOpenWeatherMapArgs] = GetFromOpenWeatherMapArgs

    def _run(self, IANA_timezone: str='Asia/Seoul', Days_from_now: int=0)-> str:
        WEATHER_API_KEY = os.getenv('OPENWEATHERMAP_API_KEY')
        region = IANA_timezone.split('/')[-1] if '/' in IANA_timezone else IANA_timezone  
        region = (requests.get(f"http://api.openweathermap.org/geo/1.0/direct?q={region}&limit={1}&appid={WEATHER_API_KEY}").json())[0]
        response = requests.get(f"https://api.openweathermap.org/data/3.0/onecall?lat={region['lat']}&lon={region['lon']}&exclude=minutely,hourly&appid={WEATHER_API_KEY}").json()

        
        if Days_from_now == 0:            
            timestamp = response['current']['dt']
            data = response['current']
            temperature = data['temp'] - 273.15
        else:
            timestamp = response['daily'][Days_from_now]['dt']
            data = response['daily'][Days_from_now]
            temperature = data['temp']['day'] - 273.15
        
        humidity = response['daily'][Days_from_now]['humidity']
        wind_speed = response['daily'][Days_from_now]['wind_speed']
        dt = datetime.datetime.fromtimestamp(timestamp, pytz.timezone(response['timezone']))  
        weather_summary = response['daily'][Days_from_now]['summary']


        if Days_from_now == 0:
            return f"timezone: {region}, current time-> year-month-day: {dt.year}-{dt.month}-{dt.day}, hour-minute: {dt.hour}-{dt.minute}, temperature: {temperature}, humidity: {humidity}, wind_speed: {wind_speed}, weather_summary: {weather_summary}."
        else:
            return f"timezone: {region}, current time-> year-month-day: {dt.year}-{dt.month}-{dt.day}, temperature: {temperature}, humidity: {humidity}, wind_speed: {wind_speed}, weather_summary: {weather_summary}."

    async def _arun(self, IANA_timezone: str, Days_from_now: int=0)-> str:
        raise NotImplementedError("Does not support async operation yet.")
