import time
from openai.error import RateLimitError

def retry_rate_limit_error(original_func):
    def wrapper(*args, **kwargs):
        while True:
            try:
                return original_func(*args, **kwargs)
            except RateLimitError as e:
                try:
                    wait_time = int(str(e).split(' Please retry after ')[1].split(' seconds. ')[0])
                except:
                    wait_time = 20
                print(f'RateLimitError -----> Will automatically retry {wait_time} seconds later.')     
                for s in range(wait_time, 0, -1):
                    print(s, end=' ')
                    time.sleep(1)
    return wrapper
 