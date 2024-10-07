import os
from abc import ABC

import redis
from redis import Redis
from dotenv import load_dotenv

class RedisInstance(ABC):
    def __init__(self, host:str='localhost', port:int=6379) -> None:
        self.client = redis.Redis(host=host, port=port)

    def get_client(self) -> Redis:
        return self.client

load_dotenv()

REDIS_HOST = os.getenv('REDIS_HOST')
REDIS_PORT = os.getenv('REDIS_PORT')

redis_instance = RedisInstance(host=REDIS_HOST, port=REDIS_PORT)
redis_client = redis_instance.get_client()