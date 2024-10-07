import pickle
import time
import logging
import sys

from model.inference import PredictionModel
from rabbitmq_config.rabbitmq_init import rabbitmq_client
from redis_config.redis_init import redis_client

from pika import BasicProperties
from pika.adapters.blocking_connection import BlockingChannel

logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s:\t%(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def run_inference(channel: BlockingChannel, method, properties: BasicProperties, body: bytes):
    print(len(body))
    item_id = pickle.loads(body)
    t = time.time()
    result = inference_model.infer(item_id)
    logger.info("Result consumer: {}. {}".format(result, time.time()-t))

    redis_client.set(properties.headers["inference_id"], pickle.dumps(result))

    channel.basic_ack(delivery_tag=method.delivery_tag)

inference_model = PredictionModel()
logger.info("Done loading model")

rabbitmq_client.basic_qos(prefetch_count=1)
rabbitmq_client.basic_consume(queue='amount_prediction_queue', on_message_callback=run_inference)

rabbitmq_client.start_consuming()