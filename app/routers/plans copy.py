from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from typing import Any, Dict

from ..planner.engine import generate_plans
from .auth import get_current_user  # adjust import if needed

router = APIRouter(prefix="/plans", tags=["plans"])

class PlansRequest(BaseModel):
  profile: Dict[str, Any]

@router.post("/recommendations")
def get_recommendations(payload: PlansRequest, user=Depends(get_current_user)):
    if not payload.profile:
        raise HTTPException(status_code=400, detail="profile is required")
    return generate_plans(payload.profile)
