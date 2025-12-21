from fastapi import Header, HTTPException
from typing import Optional, Dict
import time
from firebase_admin import auth as fb_auth


def get_current_user(authorization: Optional[str] = Header(None)) -> Dict:
    """Verify Firebase ID token from 'Authorization: Bearer <token>' and return decoded user context."""
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing Bearer token")
    token = authorization.split(" ", 1)[1]
    try:
        decoded = fb_auth.verify_id_token(token)
        return decoded
    except Exception as e:
        # tiny clock skew tolerance
        if "Token used too early" in str(e):
            time.sleep(1)
            decoded = fb_auth.verify_id_token(token)
            return decoded
        raise HTTPException(status_code=401, detail=f"Invalid token: {e}")




