REACT_REFLECTIONS = """Previous trial:
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
Jugdement: Your answer is incorrect.

Reflection: The term 'former former president' is relative and requires an understanding of the current date to determine the correct individual. I made a mistake that 'former former president' referred to the immediate predecessor of the current president. I should have first established the current date and then identified the president two terms ago, relative to the present.

Previous trial:
Query: Which one is bigger, 'Joe Biden's birth month plus Donald Trump's birth month' vs. 11?
Thought 1: I need to find out the birth months of Joe Biden and Donald Trump. I can use the GetFromYDC tool to search for this information.
Action 1: GetFromYDC(Joe Biden birth month)
Observation 1: Joe Biden was born in the month of November.
Thought 2: Now I know that Joe Biden was born in November, which is the 11th month of the year. Next, I need to find out Donald Trump's birth month. I can use the GetFromYDC tool again to search for this information.
Action 2: GetFromYDC(Donald Trump birth month)
Observation 2: Donald Trump was born in the month of June.
Thought 3: Now I know that Donald Trump was born in June, which is the 6th month of the year. So, the sum of Joe Biden's and Donald Trump's birth months is 11 + 6 = 17. Now I can compare this sum with 11.
Action 3: Python_REPL(print(17 > 11))
Observation 3: True
Jugdement: You failed to provide an answer because you exceeded the permitted number of reasoning steps. You must give an answer within 3 steps.

Reflection: In this trial, I did not manage to provide a response within the allocated time steps because I used Python_REPL tool for a simple arithmetic which is certainly what I can do without relying on any math tool. In the future trial, I will do simple math directly on my own so that I can save time slot.

Previous trial:
Query: How will be the weather in Seoul 8 days later?
Thought 1: The GetFromOpenWeatherMap tool can provide weather forecasts up to 7 days in advance, but the user is asking for the weather 8 days later, which is beyond the tool's capability. I should inform the user about this limitation.
Action: 1: Final Answer(I'm sorry, but I can only provide weather forecasts up to 7 days in advance. Please ask again closer to the date.)
Answer: I'm sorry, but I can only provide weather forecasts up to 7 days in advance. Please ask again closer to the date.
Jugdement: Your answer is incorrect.

Reflection: I correctly identified the limitations of the 'GetFromOpenWeatherMap' tool, which provides weather forecasts up to 7 days in advance, and just gave up. However, I have access to 'GetFromYDC' tool for internet searches. So I should have utilized this tool to conduct a search for the 8-day weather forecast."""

REACT_REFLECTION_HEADER = 'You have attempted to answer following question before and failed. The following reflection(s) give a plan to avoid failing to answer the question in the same way you did previously. Use them to improve your strategy of correctly answering the given question.'
