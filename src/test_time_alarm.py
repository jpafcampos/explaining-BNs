import time
import signal
import gc

# 1. Define a custom exception to catch the timeout cleanly
class TimeoutException(Exception):
    pass

# 2. Define the handler that raises the exception when time is up
def _timeout_handler(signum, frame):
    raise TimeoutException("Execution exceeded the time budget.")

def run_for_time(func, *args, timeout_sec=10, **kwargs):
    """Runs natively with a strict time budget and memory cleanup."""
    
    # Register the signal handler
    signal.signal(signal.SIGALRM, _timeout_handler)
    
    gc.collect()
    start_time = time.perf_counter()
    
    try:
        # 3. Start the countdown alarm (e.g., 1800 seconds)
        signal.alarm(timeout_sec)
        
        result = func(*args, **kwargs)
        
        # 4. If the function finishes in time, cancel the alarm IMMEDIATELY
        signal.alarm(0)
        
        if hasattr(result, '__iter__') and not isinstance(result, (list, dict, set, str)):
            result = list(result)
            
        gc.collect()
        elapsed = time.perf_counter() - start_time
        return result, elapsed, True

    except TimeoutException:
        # The alarm went off before the function finished
        print(f"\n[!] TIMEOUT: {func.__name__} aborted after {timeout_sec} seconds.")
        # We know exactly how long it took: the timeout limit
        return None, float(timeout_sec), False
        
    except Exception as e:
        print(f"\n[!] run_for_time: {func.__name__} failed with {type(e).__name__}: {e}")
        return None, (time.perf_counter() - start_time), False
        
    finally:
        # 5. Safety catch: Guarantee the alarm is turned off no matter what happens
        signal.alarm(0)
    
def long_function():
    return "Done"
    while True:
        a = 1 + 1

    return "Done"

if __name__ == "__main__":
    result, elapsed, success = run_for_time(long_function, timeout_sec=10)
    print(f"Result: {result}, Elapsed: {elapsed:.2f} seconds, Success: {success}")

