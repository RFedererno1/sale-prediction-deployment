from typing import Union

import pickle
from fastapi import APIRouter

from redis_config.redis_init import redis_client
from app.db.schemas import PendingPredictionDTO, PredictionResultDTO

router = APIRouter(prefix='/result', tags=["result"])

@router.get("/result/{inference_id}", response_model=Union[PendingPredictionDTO, PredictionResultDTO], 
            summary="return the model predictions")
async def prediction_results(inference_id:str):
    has_result = redis_client.exists(inference_id)
    if not has_result:
        return PendingPredictionDTO(inference_id=inference_id)

    result = pickle.loads(redis_client.get(inference_id))
    return PredictionResultDTO(item_id=result['item_id'], predicted_value=result['predicted_value'])