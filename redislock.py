import redis
import time

r = redis.Redis(host='localhost', port=6379, db=0)

class RedisLock:
    def __init__(self, key, timeout=10):
        self.key = key
        self.timeout = timeout
        self.acquired = False

    def __enter__(self):
        start_time = time.time()
        while time.time() - start_time < self.timeout:
            if r.setnx(self.key, 1):
                self.acquired = True
                break
            time.sleep(0.1)

        return self.acquired

    def __exit__(self, type, value, traceback):
        if self.acquired:
            r.delete(self.key)
