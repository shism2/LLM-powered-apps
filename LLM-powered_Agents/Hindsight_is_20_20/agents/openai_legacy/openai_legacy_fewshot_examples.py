OPENAI_LEGACY_REFLECTIONS = """Previous trial:
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


OPENAI_LEGACY_REFLECTION_HEADER = 'You have attempted to answer following question before and failed. The following reflection(s) give a plan to avoid failing to answer the question in the same way you did previously. Use them to improve your strategy of correctly answering the given question.'