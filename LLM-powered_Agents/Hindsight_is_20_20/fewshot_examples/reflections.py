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



OPENAI_REFLECTIONS = """Previous trial:
Query: Which one is bigger, 'World War 2 outbreak year plus Donald Trumps birth month' vs. 'U.S. Independece-day year plus Michael Jordan birth year'?
Invoking 1: Python_REPL(print(1939 + 6, 1776 + 1963))
Observation 1: 1945 3739
Invoking 2: 1945 vs. 3739
Answer: 1945 vs. 3739
Jugdement: Your answer is incorrect.

Reflection: I've successfully executed the initial step by computing the numerical values for both expressions, resulting in 1945 for the 'World War 2 outbreak year plus Donald Trump's birth month' and 3739 for the 'U.S. Independence Day year plus Michael Jordan's birth year'. However, the process was not carried through to its necessary conclusion. The final and critical step of the task is to compare the two resultant numbers, determine which one is greater, and then communicate the corresponding event as the answer. In this instance, I should explicitly state that 3739 is the larger number and thus the correct answer is 'U.S. Independence Day year plus Michael Jordan's birth year' is bigger.

Previous trial:
Query: Which one is bigger, 'Joe Biden's birth month plus Donald Trump's birth month' vs. 11?
Invoking 1: GetFromYDC(Query=Joe Biden birth month)
Observation 1: Joe Biden was born in the month of November.
Invoking 2: GetFromYDC(Query=Donald Trump birth month)
Observation 2: Donald Trump was born in the month of June.
Invoking 3: Python_REPL(print(17 > 11))
Observation 3: True
Jugdement: You failed to provide an answer because you exceeded the permitted number of reasoning steps. You must give an answer within 3 steps.

Reflection: In this trial, I did not manage to provide a response within the allocated time steps because I used Python_REPL tool for a simple arithmetic which is certainly what I can do without relying on any math tool. In the future trial, I will do simple math directly on my own so that I can save time slot.


Previous trial:
Query: Today, how old is the current Korean president?
Invoking 1: GetFromYDC(Query='current Korean president age')
Observation 1: The current president of South Korea, Yoon Suk Yeol, was born on December 18, 1960. As of today, he is 61 years old.
Invoking 2: 61 years old.
Answer: 61 years old.
Jugdement: Your answer is incorrect.

Reflection: The query necessitates a dynamic calculation of the current Korean president's age based on the current date, as ages change with the passing of time. In the future, I have to factor in the current date to provide an accurate and up-to-date response to age-related queries."""


OPENAI_PARALLEL_CALLING_REFLECTIONS = """Previous trial:
Query: Who is older? The former president of Korea vs. the MLB American League MVP of the last season. What is the result if you sum up the ages of the two?
(step 1-action) Tool invocations in parallel: [Tool 1-> YDCSearch(query=former president of South Korea name), Tool 2-> YDCSearch(query=MLB American League MVP last season name)]
(step 1-observation) Tool-invocations results in parallel:: [Tool 1-> The former president of South Korea is Park Geun-hye., Tool 2-> The American League MVP for the last season (2023) was Shohei Ohtani of the LA Angels.]
(step 2-action) Tool invocations in parallel: [Tool 1-> YDCSearch(query=Park Geun-hye age), Tool 2-> YDCSearch(query=Shohei Ohtani age)]
(step 2-observation) Tool-invocations results in parallel:: [Tool 1-> Park Geun-hye was born on February 2, 1952. As of my knowledge cutoff in 2023, she would be 71 years old., Tool 2-> Shohei Ohtani is 29 years old.]
(step 3-action) Tool invocations in parallel: Now I can answer directly without resorting to any tool.
Answer: Park Geun-hye is older. Age sum: 100 years.
Jugdement: Your answer is incorrect.

Reflection: My answer was incorrect due to an outdated result retrieved by the web search tool. The term 'former' is relative and can change over time, which means that the identity of the 'former president of South Korea' depends on the current date. To ensure accuracy in future tasks, it's crucial to establish the present context before conducting a web search. So my new strategy is to start by obtaining the current date and time using an appropriate tool. Then, I should formulate a precise search query that incorporates this information to accurately identify the 'former president' as of the most recent context. This approach will help the web search tool provide a result that reflects the current reality and avoids the ambiguity caused by time-sensitive terms like 'former.'

Previous trial:
Query: Who is older? The former president of Korea vs. the MLB American League MVP of the last season. What is the result if you sum up the ages of the two?
(step 1-action) Tool invocations in parallel: [Tool 1-> YDCSearch(query=former president of Korea), Tool 2-> YDCSearch(query=MLB American League MVP 2022)]
(step 1-observation) :: [Tool 1-> The former president of South Korea is Moon Jae-in, who served from 2017 to 2022. Prior to Moon Jae-in, the president was Park Geun-hye, who served from 2013 until her impeachment in 2017., Tool 2-> The American League MVP for the 2022 MLB season was Aaron Judge of the New York Yankees.]
(step 2-action) Tool invocations in parallel: [Tool 1-> YDCSearch(query=Moon Jae-in age), Tool 2-> YDCSearch(query=Aaron Judge age)]
(step 2-observation) :: [Tool 1-> Moon Jae-in was born on January 24, 1953. As of the time of your question, he would be 70 years old., Tool 2-> Aaron Judge is 31 years old.]
(step 3-action) Tool invocations in parallel: [Tool 1-> PythonREPL(command=print(70 + 31))]
(step 3-observation) :: [Tool 1-> 101]
(step 4-action) Tool invocations in parallel: Now I can answer directly without resorting to any tool.
Answer: The former former president of Korea is older. The sum of their age is 101.
Jugdement: Your answer is incorrect.

Reflection: I made a mistake that I interpreted the term 'last season' as the 2022 MLB season. While the assumption that the current year is 2023 was correct, I failed to consider the MLB season's timing. MLB seasons typically start in spring (March or April) and conclude in autumn (usually by the end of October). Therefore, if the current month is beyond October in the year 2023, 'last season' should refer to the 2023 MLB season, not 2022. In a future trial, I should make a plan with the fact in mind."""
