import time
from openai import RateLimitError
from functools import wraps
from typing import Optional
import asyncio

def get_waiting_time(e):
    try:
        wait_time = int(str(e).split(' Please retry after ')[1].split(' seconds. ')[0])
    except:
        wait_time = 20
    return wait_time



def retry(allowed_exceptions=(Exception,), return_message: str=''):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            retries = 0
            while True:
                try:
                    return func(*args, **kwargs)
                except allowed_exceptions as e:
                    wait_time = get_waiting_time(e)
                    print(f"RateLimitError for {func.__name__}. -----> Will automatically retry {wait_time} seconds later.")
                    for s in range(wait_time, 0, -1):
                        print(s, end=' ')
                        time.sleep(1)
                except Exception as e:
                    if return_message != '':
                        return return_message+f' The error message if {e}'
                    else:
                        raise Exception

        return wrapper
    return decorator

 