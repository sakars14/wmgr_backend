# backend/app/routers/plans.py
from __future__ import annotations

import os
from typing import Any, Dict, Optional, List

from fastapi import APIRouter, Depends, HTTPException, Header
from pydantic import BaseModel

import firebase_admin
from firebase_admin import credentials, firestore

from app.routers.auth import get_current_user
from app.planner.engine import generate_plans
from app.planner.deterministic import build_financial_blueprint
from app.llm.narrator import generate_narration, NarrationResponse

router = APIRouter(tags=["plans"])

import os
from fastapi import Header, Depends, HTTPException
from firebase_admin import auth as fb_auth

def get_ctx(authorization: str = Header(None)) -> dict:
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing token")
    token = authorization.split(" ", 1)[1]
    try:
        return fb_auth.verify_id_token(token)
    except Exception:
        raise HTTPException(status_code=401, detail="Invalid token")

def is_admin_ctx(ctx: dict) -> bool:
    # Option A: Firebase custom claim
    if ctx.get("admin") is True:
        return True

    email = (ctx.get("email") or "").strip().lower()
    raw_emails = (os.getenv("ADMIN_EMAIL") or os.getenv("ADMIN_EMAILS") or "").strip()
    allowed_emails = [e.strip().lower() for e in raw_emails.replace(";", ",").split(",") if e.strip()]
    if email and allowed_emails and email in allowed_emails:
        return True

    uid = (ctx.get("uid") or ctx.get("user_id") or "").strip()
    raw_uids = (os.getenv("ADMIN_UIDS") or "").strip()
    allowed_uids = [u.strip() for u in raw_uids.replace(";", ",").split(",") if u.strip()]
    if uid and allowed_uids and uid in allowed_uids:
        return True

    return False

def require_admin(ctx: dict = Depends(get_ctx)) -> dict:
    if not is_admin_ctx(ctx):
        raise HTTPException(status_code=403, detail="Admin only")
    return ctx


def _ensure_firebase_db():
    # Safe to call multiple times
    if not firebase_admin._apps:
        cred_path = os.getenv("FIREBASE_ADMIN_CREDENTIALS", "firebase-admin.json")
        if not os.path.exists(cred_path):
            raise RuntimeError(
                f"Firebase admin credentials not found at '{cred_path}'. "
                f"Set FIREBASE_ADMIN_CREDENTIALS or place firebase-admin.json in backend root."
            )
        firebase_admin.initialize_app(credentials.Certificate(cred_path))
    return firestore.client()


def _get_uid(user: Dict[str, Any]) -> Optional[str]:
    # auth.py returns {"uid": "..."} in your project
    return user.get("uid") or user.get("user_id") or user.get("sub")


def _require_admin(db, ctx: dict):
    """
    Accepts decoded Firebase context; allows admin via custom claim/env allowlists
    and falls back to Firestore user.isAdmin flag.
    """
    # Fast path: custom claim or env allowlists
    if is_admin_ctx(ctx):
        return

    uid = _get_uid(ctx)
    if not uid:
        raise HTTPException(status_code=401, detail="Unauthenticated")

    # Fallback: explicit flag in Firestore user doc
    doc = db.collection("users").document(uid).get()
    if doc.exists and doc.to_dict().get("isAdmin") is True:
        return

    raise HTTPException(status_code=403, detail="Admin only")


class RecommendationsRequest(BaseModel):
    # If provided and != caller uid, caller must be admin
    clientId: Optional[str] = None

    # Optional override for dev/testing (if you want to POST raw profile)
    profile: Optional[Dict[str, Any]] = None


class BlueprintRequest(BaseModel):
    # If provided and != caller uid, caller must be admin
    clientId: Optional[str] = None

    # Optional override for dev/testing (if you want to POST raw profile)
    profile: Optional[Dict[str, Any]] = None


class NarrationRequest(BaseModel):
    # If provided and != caller uid, caller must be admin
    clientId: Optional[str] = None

    # Optional override for dev/testing (if you want to POST raw profile)
    profile: Optional[Dict[str, Any]] = None


@router.post("/plans/recommendations")
def plans_recommendations(
    req: RecommendationsRequest,
    user: Dict[str, Any] = Depends(get_current_user),
    includeNarration: bool = False,
):
    db = _ensure_firebase_db()

    caller_uid = _get_uid(user)
    if not caller_uid:
        raise HTTPException(status_code=401, detail="Unauthenticated")

    target_uid = caller_uid
    if req.clientId and req.clientId != caller_uid:
        _require_admin(db, user)
        target_uid = req.clientId

    # 1) Build input profile
    if req.profile is not None:
        profile = req.profile
    else:
        snap = db.collection("clientProfiles").document(target_uid).get()
        if not snap.exists:
            raise HTTPException(status_code=404, detail=f"clientProfiles/{target_uid} not found")
        profile = snap.to_dict()

    # 2) Generate deterministic plan model
    plan_model = generate_plans(profile)
    blueprint = build_financial_blueprint(profile)

    # 3) Persist it (shared defaults to existing shared if present)
    ref = db.collection("clientPlans").document(target_uid)
    existing = ref.get()
    existing_shared = False
    if existing.exists:
        existing_shared = bool(existing.to_dict().get("shared", False))

    client_name = ""
    personal = (profile.get("personal") or {})
    if isinstance(personal, dict):
        client_name = personal.get("name", "") or ""

    ref.set(
        {
            "clientId": target_uid,
            "clientName": client_name,
            "shared": existing_shared,
            "updatedAt": firestore.SERVER_TIMESTAMP,
            "plan": plan_model,
            "blueprint": blueprint,
            "blueprintVersion": blueprint.get("meta", {}).get("blueprintVersion", "phase1-blueprint-v1"),
        },
        merge=True,
    )

    plan_response = dict(plan_model)
    plan_response["blueprint"] = blueprint
    if includeNarration:
        narration = generate_narration(blueprint, plan_model)
        narration_payload = narration.dict()
        ref.set(
            {
                "llmNarration": narration_payload,
                "narrationVersion": narration.narrationVersion,
                "narrationUpdatedAt": firestore.SERVER_TIMESTAMP,
            },
            merge=True,
        )
        plan_response["llmNarration"] = narration_payload
    return plan_response


@router.post("/plans/narration", response_model=NarrationResponse)
def plans_narration(
    req: NarrationRequest,
    user: Dict[str, Any] = Depends(get_current_user),
):
    db = _ensure_firebase_db()

    caller_uid = _get_uid(user)
    if not caller_uid:
        raise HTTPException(status_code=401, detail="Unauthenticated")

    target_uid = caller_uid
    if req.clientId and req.clientId != caller_uid:
        _require_admin(db, user)
        target_uid = req.clientId

    # 1) Build input profile
    if req.profile is not None:
        profile = req.profile
    else:
        snap = db.collection("clientProfiles").document(target_uid).get()
        if not snap.exists:
            raise HTTPException(status_code=404, detail=f"clientProfiles/{target_uid} not found")
        profile = snap.to_dict()

    ref = db.collection("clientPlans").document(target_uid)
    existing = ref.get()
    existing_data = existing.to_dict() if existing.exists else {}
    existing_shared = bool(existing_data.get("shared", False))

    # 2) Reuse stored blueprint/plan if available
    blueprint = existing_data.get("blueprint")
    plan_model = existing_data.get("plan")
    generated_plan = False
    generated_blueprint = False

    if not blueprint:
        blueprint = build_financial_blueprint(profile)
        generated_blueprint = True
    if not plan_model:
        plan_model = generate_plans(profile)
        generated_plan = True

    # 3) Generate narration
    narration = generate_narration(blueprint, plan_model)

    client_name = ""
    personal = (profile.get("personal") or {})
    if isinstance(personal, dict):
        client_name = personal.get("name", "") or ""

    update_payload: Dict[str, Any] = {
        "clientId": target_uid,
        "clientName": client_name,
        "shared": existing_shared,
        "plan": plan_model,
        "blueprint": blueprint,
        "blueprintVersion": blueprint.get("meta", {}).get("blueprintVersion", "phase1-blueprint-v1"),
        "llmNarration": narration.dict(),
        "narrationVersion": narration.narrationVersion,
        "narrationUpdatedAt": firestore.SERVER_TIMESTAMP,
    }
    if generated_plan or generated_blueprint:
        update_payload["updatedAt"] = firestore.SERVER_TIMESTAMP

    ref.set(update_payload, merge=True)
    return narration


@router.post("/plans/blueprint")
def plans_blueprint(
    req: BlueprintRequest,
    user: Dict[str, Any] = Depends(get_current_user),
):
    db = _ensure_firebase_db()

    caller_uid = _get_uid(user)
    if not caller_uid:
        raise HTTPException(status_code=401, detail="Unauthenticated")

    target_uid = caller_uid
    if req.clientId and req.clientId != caller_uid:
        _require_admin(db, user)
        target_uid = req.clientId

    if req.profile is not None:
        profile = req.profile
    else:
        snap = db.collection("clientProfiles").document(target_uid).get()
        if not snap.exists:
            raise HTTPException(status_code=404, detail=f"clientProfiles/{target_uid} not found")
        profile = snap.to_dict()

    blueprint = build_financial_blueprint(profile)

    ref = db.collection("clientPlans").document(target_uid)
    existing = ref.get()
    existing_data = existing.to_dict() if existing.exists else {}
    existing_shared = bool(existing_data.get("shared", False))

    client_name = ""
    personal = (profile.get("personal") or {})
    if isinstance(personal, dict):
        client_name = personal.get("name", "") or ""

    ref.set(
        {
            "clientId": target_uid,
            "clientName": client_name,
            "shared": existing_shared,
            "updatedAt": firestore.SERVER_TIMESTAMP,
            "blueprint": blueprint,
            "blueprintVersion": blueprint.get("meta", {}).get("blueprintVersion", "phase1-blueprint-v1"),
        },
        merge=True,
    )

    return blueprint


# -------------------------
# Admin endpoints for UI
# -------------------------

@router.get("/admin/plans")
def admin_plans(user=Depends(get_current_user)):
    db = _ensure_firebase_db()
    uid = _get_uid(user)
    if not uid:
        raise HTTPException(status_code=401, detail="Unauthenticated")
    _require_admin(db, user)

    # Best-effort ordering; if updatedAt isn't indexed yet, fallback to stream()
    plans_ref = db.collection("clientPlans")
    try:
        snaps = plans_ref.order_by("updatedAt", direction=firestore.Query.DESCENDING).limit(200).stream()
    except Exception:
        snaps = plans_ref.limit(200).stream()

    out: List[Dict[str, Any]] = []
    for s in snaps:
        d = s.to_dict() or {}
        out.append(
            {
                "clientId": s.id,
                "clientName": d.get("clientName", ""),
                "shared": bool(d.get("shared", False)),
                "updatedAt": d.get("updatedAt"),
            }
        )
    return out


@router.get("/admin/plans/{client_id}")
def admin_get_plan(client_id: str, user: Dict[str, Any] = Depends(get_current_user)):
    db = _ensure_firebase_db()
    uid = _get_uid(user)
    if not uid:
        raise HTTPException(status_code=401, detail="Unauthenticated")
    _require_admin(db, user)

    snap = db.collection("clientPlans").document(client_id).get()
    if not snap.exists:
        raise HTTPException(status_code=404, detail=f"clientPlans/{client_id} not found")
    d = snap.to_dict() or {}
    return {
        "clientId": client_id,
        "clientName": d.get("clientName", ""),
        "shared": bool(d.get("shared", False)),
        "plan": d.get("plan"),
        "updatedAt": d.get("updatedAt"),
    }


class ShareRequest(BaseModel):
    shared: bool


@router.post("/admin/plans/{client_id}/share")
def admin_share_plan(
    client_id: str,
    req: ShareRequest,
    user: Dict[str, Any] = Depends(get_current_user),
):
    db = _ensure_firebase_db()
    uid = _get_uid(user)
    if not uid:
        raise HTTPException(status_code=401, detail="Unauthenticated")
    _require_admin(db, user)

    ref = db.collection("clientPlans").document(client_id)
    ref.set(
        {
            "shared": bool(req.shared),
            "updatedAt": firestore.SERVER_TIMESTAMP,
        },
        merge=True,
    )
    return {"ok": True, "clientId": client_id, "shared": bool(req.shared)}


# -------------------------
# User-facing plan visibility
# -------------------------

@router.get("/plans/shared/me")
def plans_shared_me(user: Dict[str, Any] = Depends(get_current_user)):
    db = _ensure_firebase_db()
    uid = _get_uid(user)
    if not uid:
        raise HTTPException(status_code=401, detail="Unauthenticated")

    snap = db.collection("clientPlans").document(uid).get()
    if not snap.exists:
        return {"hasPlan": False, "shared": False}

    data = snap.to_dict() or {}
    return {
        "hasPlan": True,
        "shared": bool(data.get("shared", False)),
    }


@router.get("/plans/my")
def plans_my(user: Dict[str, Any] = Depends(get_current_user)):
    db = _ensure_firebase_db()
    uid = _get_uid(user)
    if not uid:
        raise HTTPException(status_code=401, detail="Unauthenticated")

    snap = db.collection("clientPlans").document(uid).get()
    if not snap.exists:
        raise HTTPException(status_code=404, detail="Plan not generated yet")

    data = snap.to_dict() or {}
    if not data.get("shared", False):
        raise HTTPException(status_code=403, detail="Plan not shared yet")

    return {"plan": data.get("plan"), "clientId": uid, "clientName": data.get("clientName", "")}
