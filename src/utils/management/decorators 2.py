import os
import time
import functools
import logging
from datetime import datetime
from .file_manager import get_git_project_root

#################
#   Variables   #
#################

# Paths
ROOT = get_git_project_root(os.path.dirname(os.path.realpath(__file__)))

# logger 
logger = logging.getLogger(__name__)


##################
#   Decorators   #
##################

def timer(_func = None):
    """
    Custom timer decorator.
    """
    def decorator_timer(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.perf_counter()
            value = func(*args, **kwargs)
            end_time = time.perf_counter()
            run_time = end_time - start_time
            print(f"Finished {func.__name__!r} in {run_time:.4f} secs")
            return value
        return wrapper
    if _func is None: return decorator_timer
    else: return decorator_timer(_func)


def log(_func = None, include_timer = False):
    """
    Custom logging decorator for programs centralized logger
    """
    def decorator_log(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                # infomation for logging
                args_repr = [repr(a) for a in args]
                kwargs_repr = [f"{k}={v!r}" for k, v in kwargs.items()]
                signature = ", ".join(args_repr + kwargs_repr)
                logger.info(f"function {func.__name__} called with args {signature}")

                # execute function
                start_time = time.perf_counter()
                res = func(*args, *kwargs)
                end_time = time.perf_counter()
                run_time = end_time - start_time # execution time for logging

                if include_timer: logger.info(f"function {func.__name__} finished with execution time {run_time}")
                else: logger.info(f"function {func.__name__} finished")
                return res
            
            except Exception as e:
                logger.exception(f"Exception raised in {func.__name__}. exception: {str(e)}")
                raise e
        return wrapper

    if _func is None: return decorator_log
    else: return decorator_log(_func)

