from diskcache import Cache
from pathlib import Path
from app.utils.settings import CACHE_DIR

Path(CACHE_DIR).mkdir(parents=True, exist_ok=True)
cache = Cache(CACHE_DIR)

def get(key):
    return cache.get(key)

def set(key, value, expire=24*3600):
    cache.set(key, value, expire=expire)
