from __future__ import annotations

import os
import time
from typing import Any, Dict, Optional, Tuple

from fastapi import Depends, HTTPException, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from firebase_admin import auth as fb_auth
from google.cloud import firestore

bearer_scheme = HTTPBearer(auto_error=False)
db = firestore.Client()


def _csv_set(val: str) -> set[str]:
    raw = (val or "").replace(";", ",").replace("\n", ",")
    return {item.strip().lower() for item in raw.split(",") if item.strip()}


def get_user_ctx(
    request: Request,
    creds: Optional[HTTPAuthorizationCredentials] = Depends(bearer_scheme),
) -> Dict[str, Any]:
    token: Optional[str] = None

    if creds and creds.credentials:
        token = creds.credentials
    else:
        auth_header = request.headers.get("Authorization") or request.headers.get("authorization")
        if auth_header:
            raw = auth_header.strip()
            token = raw.split(" ", 1)[1].strip() if raw.lower().startswith("bearer ") else raw

    if not token:
        raise HTTPException(status_code=401, detail="Missing Bearer token")

    try:
        return fb_auth.verify_id_token(token)
    except Exception as e:
        if "Token used too early" in str(e):
            time.sleep(1)
            return fb_auth.verify_id_token(token)
        raise HTTPException(status_code=401, detail=f"Invalid token: {e}")


def get_uid(ctx: Dict[str, Any] = Depends(get_user_ctx)) -> str:
    uid = ctx.get("uid")
    if not uid:
        raise HTTPException(status_code=401, detail="Unauthenticated")
    return uid


def is_admin(ctx: Dict[str, Any]) -> Tuple[bool, str]:
    email = (ctx.get("email") or "").strip().lower()
    uid = ctx.get("uid") or ctx.get("user_id")

    if ctx.get("admin") is True:
        return True, "claim:admin"

    if uid:
        try:
            admin_doc = db.collection("admins").document(uid).get()
            if admin_doc.exists:
                admin_data = admin_doc.to_dict() or {}
                enabled = admin_data.get("enabled")
                if enabled is None or enabled is True:
                    return True, "firestore:admins"

            user_doc = db.collection("users").document(uid).get()
            if user_doc.exists:
                data = user_doc.to_dict() or {}
                if data.get("isAdmin") is True:
                    return True, "firestore:users.isAdmin"
                roles = data.get("roles") or {}
                if isinstance(roles, dict) and roles.get("admin") is True:
                    return True, "firestore:roles.admin"
        except Exception:
            pass

    email_allow = _csv_set(
        ",".join(
            filter(
                None,
                [
                    os.getenv("ADMIN_EMAIL_ALLOWLIST"),
                    os.getenv("ADMIN_EMAILS"),
                    os.getenv("ADMIN_EMAIL"),
                ],
            )
        )
    )
    uid_allow = _csv_set(
        ",".join(
            filter(
                None,
                [
                    os.getenv("ADMIN_UID_ALLOWLIST"),
                    os.getenv("ADMIN_UIDS"),
                    os.getenv("ADMIN_UID"),
                ],
            )
        )
    )
    if uid and uid.lower() in uid_allow:
        return True, "allowlist:uid"
    if email and email in email_allow:
        return True, "allowlist:email"

    return False, "none"


def require_admin(ctx: Dict[str, Any] = Depends(get_user_ctx)) -> Dict[str, Any]:
    allowed, reason = is_admin(ctx)
    if not allowed:
        email = (ctx.get("email") or "").strip().lower()
        uid = ctx.get("uid") or ctx.get("user_id")
        raise HTTPException(
            status_code=403,
            detail=f"Admin only (reason={reason}, email={email}, uid={uid})",
        )
    return ctx
