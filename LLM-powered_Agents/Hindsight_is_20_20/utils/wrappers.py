import time
from openai import RateLimitError
from functools import wraps

# def retry_rate_limit_error(original_func):
#     def wrapper(*args, **kwargs):
#         while True:
#             try:
#                 return original_func(*args, **kwargs)
#             except RateLimitError as e:
#                 try:
#                     wait_time = int(str(e).split(' Please retry after ')[1].split(' seconds. ')[0])
#                 except:
#                     wait_time = 20
#                 print(f'RateLimitError -----> Will automatically retry {wait_time} seconds later.')     
#                 for s in range(wait_time, 0, -1):
#                     print(s, end=' ')
#                     time.sleep(1)
#     return wrapper


def get_waiting_time(e):
    try:
        wait_time = int(str(e).split(' Please retry after ')[1].split(' seconds. ')[0])
    except:
        wait_time = 20
    return wait_time

def retry(allowed_exceptions=(Exception,)):
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
        return wrapper
    return decorator

 