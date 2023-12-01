OPENAI_PARALLEL_REFLECTIONS = """Previous trial:
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


OPENAI_PARALLEL_REFLECTION_HEADER = 'You have attempted to answer following question before and failed. The following reflection(s) give a plan to avoid failing to answer the question in the same way you did previously. Use them to improve your strategy of correctly answering the given question.'