from typing import Dict
from pydantic import BaseModel

class PendingPredictionDTO(BaseModel):
    inference_id: str

class PredictionResultDTO(BaseModel):
    item_id: int
    predicted_value: Dict[int, int]