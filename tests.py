#!~/projects/ga/venv/bin/ python3

'''testing script'''
import time
from functools import wraps

import numpy as np

from main import individual, min_func

def timeit(func):
    @wraps(func)
    def timeit_wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        total_time = end_time - start_time
        print(f'Function {func.__name__}{args} {kwargs} Took {total_time:.4f} seconds')
        return result
    return timeit_wrapper

if __name__ == "__main__":
    
    X = np.random.randint(0, 255, size=(4 * 6))
    IND = individual(X)
    timeit(min_func(X, X + np.random.randint(0, 10)))


