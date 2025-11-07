from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field

class Leg(BaseModel):
    exchange: str
    symbol: str
    qty: int = Field(gt=0)
    product: str = "CNC"
    priceType: str = "MARKET"   # MARKET | LIMIT
    limitPrice: Optional[float] = None

class CreateOrderGroupRequest(BaseModel):
    bucketId: str
    legs: List[Leg]
    idempotencyKey: Optional[str] = None

class OrderGroupResponse(BaseModel):
    groupId: str
    status: str
    legs: List[Dict[str, Any]]
