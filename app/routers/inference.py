import uuid

import pickle
from fastapi import APIRouter
from pika import BasicProperties

from rabbitmq_config.rabbitmq_init import rabbitmq_client
from app.db.schemas import PendingPredictionDTO

router = APIRouter(prefix='/inference', tags=["inference"])

@router.post("/inference", response_model=PendingPredictionDTO, summary="predict total amount of products sold in every shop")
async def predict(item_id: int):
    inference_id = str(uuid.uuid4())

    rabbitmq_client.basic_publish(
        exchange='',
        routing_key='amount_prediction_queue',
        body=pickle.dumps(item_id),
        properties=BasicProperties(headers={'inference_id': inference_id})
    )

    return PendingPredictionDTO(inference_id=inference_id)