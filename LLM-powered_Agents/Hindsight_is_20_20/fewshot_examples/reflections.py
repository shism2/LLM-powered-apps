REFLECTIONS = """
Previous Trial:
Query: Who is the former former president of Korea? Where was the inauguration ceremony of the former former president of Korea held?
Thought 1: To answer this question, I need to find out who the former former president of Korea is and where their inauguration ceremony was held. I will use the GetFromYDC tool to search for this information.
Action: 1: GetFromYDC(former former president of Korea)
Observation 1: The former president of South Korea is Moon Jae-In, who served from 2017 to 2022.
Thought 2: Now that I know the former former president of Korea is Moon Jae-In, I need to find out where his inauguration ceremony was held. I will use the GetFromYDC tool again to search for this information.
Action: 2: GetFromYDC(Moon Jae-In inauguration ceremony location)
Observation 2: The inauguration ceremony for Moon Jae-In took place at the National Assembly building in Seoul, South Korea.
Thought 3: I know what to respond
Action: 3: Final Answer(The former former president of Korea is Moon Jae-In and his inauguration ceremony was held at the National Assembly building in Seoul, South Korea.)
Answer: The former former president of Korea is Moon Jae-In and his inauguration ceremony was held at the National Assembly building in Seoul, South Korea.
Jugdement: Your answer is incorrect. The correct answer is "The former former president of Korea is Park Geun-hye. Her inaugration ceremony was held at the National Assembly building in Seoul, South Korea."

Reflection: The term 'former former president' is relative and requires an understanding of the current date to determine the correct individual. I made a mistake that 'former former president' referred to the immediate predecessor of the current president. I should have first established the current date and then identified the president two terms ago, relative to the present.

Previous Trial:
Query: How will be the weather in Seoul 8 days later?
Thought 1: The GetFromOpenWeatherMap tool can provide weather forecasts up to 7 days in advance, but the user is asking for the weather 8 days later, which is beyond the tool's capability. I should inform the user about this limitation.
Action: 1: Final Answer(I'm sorry, but I can only provide weather forecasts up to 7 days in advance. Please ask again closer to the date.)
Answer: I'm sorry, but I can only provide weather forecasts up to 7 days in advance. Please ask again closer to the date.

Reflection: I correctly identified the limitations of the 'GetFromOpenWeatherMap' tool, which provides weather forecasts up to 7 days in advance, and just gave up. However, I have access to 'GetFromYDC' tool for internet searches. So I should have utilized this tool to conduct a search for the 8-day weather forecast.
"""

REFLECTION_HEADER = 'You have attempted to answer following question before and failed. The following reflection(s) give a plan to avoid failing to answer the question in the same way you did previously. Use them to improve your strategy of correctly answering the given question.\n'