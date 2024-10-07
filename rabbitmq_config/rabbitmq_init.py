import os
from abc import ABC
from typing import Any

import pika
from pika import PlainCredentials
from pika.adapters.blocking_connection import BlockingChannel
from dotenv import load_dotenv

class RabbitMQInstance(ABC):
    def __init__(self, host:str='localhost', port:int=5672, username:str='root', password:str='123456') -> None:
        self.credential = pika.PlainCredentials(username=username, password=password)
        self.param = pika.ConnectionParameters(
            host=host,
            port=port,
            credentials=self.credential,
            heartbeat=0,
        )
        self.connection = pika.BlockingConnection(self.param)
        self.client = self.get_client()

    def get_client(self) -> BlockingChannel:
        return self.connection.channel()
    
    def declare_queue(self, name:str='amount_prediction_queue') -> Any:
        return self.client.queue_declare(queue=name)

load_dotenv()

RABBITMQ_HOST = os.getenv('RABBITMQ_HOST')
RABBITMQ_PORT = os.getenv('RABBITMQ_PORT')
RABBITMQ_USERNAME = os.getenv('RABBITMQ_USERNAME')
RABBITMQ_PASSWORD = os.getenv('RABBITMQ_PASSWORD')

rabbitmq_instance = RabbitMQInstance(host=RABBITMQ_HOST, port=RABBITMQ_PORT, 
                                     username=RABBITMQ_USERNAME, password=RABBITMQ_PASSWORD)
rabbitmq_client = rabbitmq_instance.get_client()
rabbitmq_instance.declare_queue()