from datetime import datetime

from pydantic import BaseModel


class OrderInput(BaseModel):
    order_purchase_timestamp: datetime
    order_estimated_delivery_date: datetime

    total_items: float
    total_price: float
    total_freight: float
    total_payment_value: float
    number_of_payments: float
    max_installments: float
    seller_count: float
    distance_km: float

    customer_state: str
    seller_state: str
    
class PredictionResponse(BaseModel):
    prediction: int
    probability: float
    model_version: str

class BatchOrderInput(BaseModel):
    orders: list[OrderInput]


class BatchPredictionResponse(BaseModel):
    predictions: list[PredictionResponse]
