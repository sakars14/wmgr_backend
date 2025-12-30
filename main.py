# apps/api/main.py

from fastapi.responses import RedirectResponse
import os
import re
import uuid
import json
import time
import base64
import hmac, hashlib
import razorpay
from datetime import timedelta
from typing import Dict
from urllib.parse import quote, urlencode
from datetime import datetime, time as dtime
from zoneinfo import ZoneInfo

from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"), override=True)
import firebase_admin
from firebase_admin import credentials, auth as fb_auth
from google.cloud import firestore

from adapters.zerodha_adapter import ZerodhaAdapter
from models import CreateOrderGroupRequest, OrderGroupResponse
from market_time import choose_order_mode



# ---------- Boot ----------


# FastAPI app (create ONCE)
app = FastAPI(title="WMGR API")



# Razorpay details
client = razorpay.Client(auth=(os.environ["RAZORPAY_KEY_ID"], os.environ["RAZORPAY_KEY_SECRET"]))

#PLAN_PRICE_PAISE = {
#    "standard": 19900,  # ₹199 → paise
#    "pro":      49900,  # ₹499
#    "max":      99900   # ₹999
#}

# CORS: allow both localhost & 127.0.0.1 plus explicit APP_BASE_URL
_default_webs = ["http://127.0.0.1:3000", "http://localhost:3000", "https://wmgr-web.vercel.app"]
_app_base = os.getenv("APP_BASE_URL")
allow_origins = _default_webs if not _app_base else list(set(_default_webs + [_app_base]))

app.add_middleware(
    CORSMiddleware,
    allow_origins=allow_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Firebase Admin
if not firebase_admin._apps:
    cred_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    if not cred_path or not os.path.exists(cred_path):
        raise RuntimeError("Missing GOOGLE_APPLICATION_CREDENTIALS or file not found.")
    cred = credentials.Certificate(cred_path)
    firebase_admin.initialize_app(cred, {"projectId": os.getenv("FIREBASE_PROJECT_ID")})

# Firestore client
db = firestore.Client()

from app.deps.authz import get_user_ctx, get_uid, require_admin, is_admin
from app.routers import plans
from app.routers import model_buckets
app.include_router(plans.router)
app.include_router(model_buckets.router)

# ---------- Helpers ----------
def load_buckets():
    """Read demo buckets from seeds/buckets.json."""
    p = os.path.join(os.path.dirname(__file__), "seeds", "buckets.json")
    if not os.path.exists(p):
        return {"buckets": []}
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)["buckets"]


def init_firebase():
    if firebase_admin._apps:
        return  # already initialized (hot restarts)

    path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
    json_blob = os.environ.get("FIREBASE_ADMIN_JSON")  # optional fallback

    if path and os.path.exists(path):
        cred = credentials.Certificate(path)
    elif json_blob:
        cred = credentials.Certificate(json.loads(json_blob))
    else:
        raise RuntimeError("Missing GOOGLE_APPLICATION_CREDENTIALS or FIREBASE_ADMIN_JSON")

    firebase_admin.initialize_app(cred)

init_firebase()


# --- env helpers ---
RZP_KEY_ID = os.getenv("RAZORPAY_KEY_ID", "")
RZP_KEY_SECRET = os.getenv("RAZORPAY_KEY_SECRET", "")
RZP_WEBHOOK_SECRET = os.getenv("RAZORPAY_WEBHOOK_SECRET", "")
BILLING_CCY = os.getenv("BILLING_CURRENCY", "INR")

def _rzp_client():
    if not (RZP_KEY_ID and RZP_KEY_SECRET):
        raise HTTPException(500, "Razorpay keys missing")
    return razorpay.Client(auth=(RZP_KEY_ID, RZP_KEY_SECRET))

def _now_ist():
    return datetime.now(ZoneInfo("Asia/Kolkata"))



# ---------- Probes ----------
@app.get("/")
def root():
    return {"ok": True, "service": "wmgr-api", "cors": allow_origins}

@app.get("/health")
def health():
    return {"ok": True}

@app.get("/dev/whoami")
def whoami(ctx: Dict = Depends(get_user_ctx)):
    is_admin_flag, reason = is_admin(ctx)
    return {
        "uid": ctx.get("uid"),
        "email": ctx.get("email"),
        "is_admin": is_admin_flag,
        "admin_reason": reason,
    }

@app.post("/dev/make-me-admin")
def make_me_admin(ctx: Dict = Depends(get_user_ctx)):
    if os.getenv("ALLOW_DEV_ADMIN_BOOTSTRAP") != "1":
        raise HTTPException(status_code=403, detail="Admin bootstrap disabled")

    uid = ctx.get("uid")
    email = ctx.get("email")
    if not uid:
        raise HTTPException(status_code=401, detail="Unauthenticated")

    db.collection("admins").document(uid).set(
        {
            "enabled": True,
            "email": email,
            "createdAt": firestore.SERVER_TIMESTAMP,
        }
    )
    return {"ok": True, "uid": uid, "email": email}

@app.get("/market/status")
def market_status():
    """Simple market open/close check for NSE cash (IST 09:15–15:30, Mon–Fri)."""
    now = datetime.now(ZoneInfo("Asia/Kolkata"))
    is_open = (now.weekday() < 5) and (dtime(9, 15) <= now.time() <= dtime(15, 30))
    return {"isOpen": is_open, "nowIst": now.isoformat()}

# ---------- Buckets ----------
@app.get("/buckets")
def list_buckets(uid: str = Depends(get_uid)):
    return {"buckets": load_buckets()}

@app.get("/buckets/{bucket_id}")
def bucket_detail(bucket_id: str, uid: str = Depends(get_uid)):
    """
    Return a single bucket and enrich each leg with live price (ltp) if Zerodha works.
    """
    bucket = None
    for b in load_buckets():
        if b["id"] == bucket_id:
            bucket = b
            break

    if bucket is None:
        raise HTTPException(404, "Bucket not found")

    adapter = ZerodhaAdapter(db)
    legs = bucket.get("legs", [])

    priced_legs = []
    price_error: str | None = None

    for leg in legs:
        ex = leg.get("exchange", "NSE")
        sym = leg["symbol"]
        try:
            ltp = adapter.get_ltp(uid, ex, sym)
        except Exception as e:
            ltp = None
            if price_error is None:
                price_error = str(e)
        priced_legs.append({**leg, "ltp": ltp})

    result = {
        **bucket,
        "legs": priced_legs,
        "items": priced_legs,  # for the frontend mapping
    }
    if price_error:
        result["priceError"] = price_error

    return result

# ---------- Zerodha helpers ----------
def _b64url_encode(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def _b64url_decode(val: str) -> bytes:
    padded = val + "=" * (-len(val) % 4)
    return base64.urlsafe_b64decode(padded.encode("ascii"))


def _get_zerodha_state_secret() -> str:
    return os.getenv("ZERODHA_STATE_SECRET") or os.getenv("ZERODHA_API_SECRET") or ""


def _sanitize_return_to(value: str | None) -> str | None:
    if not value:
        return None
    if "://" in value:
        return None
    if not value.startswith("/"):
        return None
    return value


def _make_zerodha_state(uid: str, return_to: str | None) -> str:
    payload = {"uid": uid}
    if return_to:
        payload["returnTo"] = return_to
    payload_json = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    payload_b64 = _b64url_encode(payload_json)
    secret = _get_zerodha_state_secret()
    if not secret:
        return payload_b64
    sig = hmac.new(secret.encode("utf-8"), payload_b64.encode("utf-8"), hashlib.sha256).hexdigest()
    return f"{payload_b64}.{sig}"


def _parse_zerodha_state(state: str):
    if not state:
        return None
    if "." not in state:
        try:
            payload = json.loads(_b64url_decode(state))
            return payload if isinstance(payload, dict) else None
        except Exception:
            return None
    payload_b64, sig = state.rsplit(".", 1)
    secret = _get_zerodha_state_secret()
    if not secret:
        return None
    expected = hmac.new(secret.encode("utf-8"), payload_b64.encode("utf-8"), hashlib.sha256).hexdigest()
    if not hmac.compare_digest(expected, sig):
        return None
    try:
        payload = json.loads(_b64url_decode(payload_b64))
        return payload if isinstance(payload, dict) else None
    except Exception:
        return None


def _ts_to_iso(value):
    if not value:
        return None
    if isinstance(value, datetime):
        return value.isoformat()
    if hasattr(value, "to_datetime"):
        try:
            return value.to_datetime().isoformat()
        except Exception:
            return None
    if hasattr(value, "isoformat"):
        try:
            return value.isoformat()
        except Exception:
            return None
    return None

# ---------- Zerodha OAuth ----------
@app.get("/auth/zerodha/login")
def zerodha_login(request: Request, uid: str = Depends(get_uid)):
    from kiteconnect import KiteConnect

    api_key = os.getenv("ZERODHA_API_KEY")
    base_redirect = os.getenv("ZERODHA_REDIRECT_URL")
    if not api_key:
        raise HTTPException(500, "ZERODHA_API_KEY missing")
    if not base_redirect:
        raise HTTPException(500, "ZERODHA_REDIRECT_URL missing")

    k = KiteConnect(api_key=api_key)

    return_to = _sanitize_return_to(request.query_params.get("returnTo"))
    redirect_params = {"uid": uid}
    if return_to:
        redirect_params["returnTo"] = return_to
    sep = "&" if "?" in base_redirect else "?"
    redirect_with_uid = base_redirect + sep + urlencode(redirect_params, safe="/")
    state = _make_zerodha_state(uid, return_to)
    login_url = (
        k.login_url()
        + "&redirect_url=" + quote(redirect_with_uid, safe="")
        + "&state=" + quote(state, safe="")
    )
    return {"loginUrl": login_url}

# Convenience: redirect the browser straight to Kite login (handy for testing from address bar)
@app.get("/auth/zerodha/login/redirect")
def zerodha_login_redirect(request: Request, uid: str = Depends(get_uid)):
    data = zerodha_login(request=request, uid=uid)
    return RedirectResponse(url=data["loginUrl"], status_code=302)

@app.get("/auth/zerodha/callback")
def zerodha_callback(request: Request):
    from kiteconnect import KiteConnect

    params = dict(request.query_params)
    request_token = params.get("request_token")
    uid = params.get("uid")
    return_to = params.get("returnTo")
    state = params.get("state")
    if state:
        parsed = _parse_zerodha_state(state)
        if parsed:
            uid = parsed.get("uid") or uid
            return_to = parsed.get("returnTo") or return_to
        elif not uid and "." not in state:
            uid = state
    return_to = _sanitize_return_to(return_to) or "/buckets"

    if not request_token:
        return {"ok": False, "stage": "callback", "error": "Missing request_token", "params": params}
    if not uid:
        return {"ok": False, "stage": "callback", "error": "Missing uid/state", "params": params}

    try:
        k = KiteConnect(api_key=os.getenv("ZERODHA_API_KEY"))
        data = k.generate_session(request_token, api_secret=os.getenv("ZERODHA_API_SECRET"))
        access_token = data.get("access_token")
        if not access_token:
            return {"ok": False, "stage": "generate_session", "error": "No access token", "data": data}

        email = None
        try:
            user_record = fb_auth.get_user(uid)
            email = user_record.email
        except Exception:
            email = None

        db.collection("zerodhaConnections").document(uid).set(
            {
                "uid": uid,
                "email": email,
                "kiteUserId": data.get("user_id"),
                "accessToken": access_token,
                "connected": True,
                "connectedAt": firestore.SERVER_TIMESTAMP,
                "updatedAt": firestore.SERVER_TIMESTAMP,
            },
            merge=True,
        )

        db.collection("users").document(uid).collection("brokerLinks").document("zerodha").set(
            {"accessToken": access_token, "createdAt": firestore.SERVER_TIMESTAMP}
        )

        # dY`? Redirect the browser to your frontend buckets page
        app_base = os.getenv("APP_BASE_URL", "http://127.0.0.1:3000")
        return RedirectResponse(url=f"{app_base}{return_to}", status_code=302)

    except Exception as e:
        # if anything goes wrong, show structured error
        return {"ok": False, "stage": "exception", "error": str(e), "params": params}

# --- Zerodha aliases so your Render URL matches Zerodha console entries exactly ---
@app.get("/zerodha/login")
def zerodha_login_alias(request: Request, uid: str = Depends(get_uid)):
    return zerodha_login(request=request, uid=uid)

@app.get("/zerodha/callback")
def zerodha_callback_alias(request: Request):
    return zerodha_callback(request)

def _zerodha_status(ctx: Dict):
    uid = ctx.get("uid")
    if not uid:
        raise HTTPException(status_code=401, detail="Unauthenticated")

    doc_ref = db.collection("zerodhaConnections").document(uid)
    doc = doc_ref.get()
    data = doc.to_dict() if doc.exists else {}

    adapter = ZerodhaAdapter(db)
    connected = False
    reason = "no_record"
    error = None

    try:
        adapter.margins(uid)
        connected = True
        reason = "ok"
    except Exception as e:
        connected = False
        error = str(e)
        reason = "token_invalid" if doc.exists else "no_record"

    if connected:
        if doc.exists:
            doc_ref.set(
                {"connected": True, "updatedAt": firestore.SERVER_TIMESTAMP, "lastValidatedAt": firestore.SERVER_TIMESTAMP},
                merge=True,
            )
        else:
            try:
                token = adapter._get_access_token(uid)
            except Exception:
                token = None
            if token:
                doc_ref.set(
                    {
                        "uid": uid,
                        "email": ctx.get("email"),
                        "accessToken": token,
                        "connected": True,
                        "connectedAt": firestore.SERVER_TIMESTAMP,
                        "updatedAt": firestore.SERVER_TIMESTAMP,
                        "lastValidatedAt": firestore.SERVER_TIMESTAMP,
                    },
                    merge=True,
                )
    else:
        if doc.exists:
            doc_ref.set(
                {"connected": False, "updatedAt": firestore.SERVER_TIMESTAMP, "lastValidatedAt": firestore.SERVER_TIMESTAMP},
                merge=True,
            )

    out = {"connected": connected, "reason": reason}
    if data:
        kite_user_id = data.get("kiteUserId")
        updated_at = _ts_to_iso(data.get("updatedAt"))
        connected_at = _ts_to_iso(data.get("connectedAt"))
        if kite_user_id:
            out["kiteUserId"] = kite_user_id
        if updated_at:
            out["updatedAt"] = updated_at
        if connected_at:
            out["connectedAt"] = connected_at
    if not connected and error:
        out["error"] = error
    return out

@app.get("/auth/zerodha/status")
@app.get("/zerodha/status")
def zerodha_status(ctx: Dict = Depends(get_user_ctx)):
    return _zerodha_status(ctx)

@app.post("/zerodha/quotes")
def zerodha_quotes(body: Dict, ctx: Dict = Depends(get_user_ctx)):
    instruments = (body or {}).get("instruments") or []
    if not isinstance(instruments, list) or not instruments:
        raise HTTPException(status_code=400, detail="Missing instruments")
    adapter = ZerodhaAdapter(db)
    try:
        prices = adapter.get_quotes(ctx["uid"], instruments)
    except Exception as e:
        raise HTTPException(status_code=401, detail=f"Zerodha not connected: {e}")
    return {"prices": prices}

@app.get("/admin/zerodha-status")
def admin_zerodha_status(uids: str = "", _: Dict = Depends(require_admin)):
    raw = [u.strip() for u in (uids or "").split(",") if u.strip()]
    if not raw:
        return {"statuses": {}}

    doc_refs = [db.collection("zerodhaConnections").document(uid) for uid in raw]
    statuses = {}
    for doc in db.get_all(doc_refs):
        if not doc.exists:
            statuses[doc.id] = {"connected": False}
            continue
        data = doc.to_dict() or {}
        entry = {"connected": bool(data.get("connected"))}
        updated_at = _ts_to_iso(data.get("updatedAt"))
        if updated_at:
            entry["updatedAt"] = updated_at
        statuses[doc.id] = entry

    for uid in raw:
        if uid not in statuses:
            statuses[uid] = {"connected": False}

    return {"statuses": statuses}

# ---------- Orders (Plan-gated + AMO buffer) ----------
PLAN_LIMIT = {"none": 0, "standard": 5, "pro": 8, "max": 10}

def _derive_rank(bucket_id: str) -> int:
    """Extract a numeric order (1..N) from bucket id/name for gating."""
    m = re.search(r"(\d+)", bucket_id or "")
    return int(m.group(1)) if m else 999

@app.post("/orders/group", response_model=OrderGroupResponse)
def create_order_group(payload: CreateOrderGroupRequest, auth: dict = Depends(get_user_ctx)):
    """
    Places a group of orders. Enforces plan gating by bucket rank and adds AMO-LIMIT buffer
    price when outside market hours (choose_order_mode decides the variety/order_type).
    """
    uid = auth["uid"]
    user_plan = (auth.get("plan") or "none").lower()
    allowed = PLAN_LIMIT.get(user_plan, 0)

    # Plan gating by bucket "rank" (first 5/8/10 buckets etc.)
    bucket_rank = _derive_rank(payload.bucketId or "")
    if bucket_rank > allowed:
        raise HTTPException(403, f"Bucket locked for plan '{user_plan}'")

    # Idempotency
    if payload.idempotencyKey:
        q = (db.collection("order_groups")
               .where("uid", "==", uid)
               .where("idempotencyKey", "==", payload.idempotencyKey)
               .limit(1)
               .stream())
        existing = next(q, None)
        if existing:
            d = existing.to_dict()
            legs = [l.to_dict() for l in db.collection("order_legs")
                                      .where("groupId", "==", d["groupId"]).stream()]
            return OrderGroupResponse(groupId=d["groupId"], status=d["status"], legs=legs)

    group_id = uuid.uuid4().hex
    db.collection("order_groups").document(group_id).set({
        "groupId": group_id,
        "uid": uid,
        "bucketId": payload.bucketId,
        "status": "pending",
        "idempotencyKey": payload.idempotencyKey or None,
        "createdAt": firestore.SERVER_TIMESTAMP
    })

    adapter = ZerodhaAdapter(db)
    legs_resp, errors = [], False
    amo_buffer = float(os.getenv("AMO_LIMIT_BUFFER", "1.01"))  # 1% above LTP by default

    for leg in payload.legs:
        leg_id = uuid.uuid4().hex

        # Decide variety/order_type (e.g., AMO+LIMIT outside hours)
        variety, order_type = choose_order_mode(leg.priceType)

        # Decide price: if LIMIT and no/zero price provided, set from LTP with buffer
        price = leg.limitPrice
        if order_type == "LIMIT" and (price is None or price <= 0):
            try:
                ltp = adapter.get_ltp(uid, leg.exchange, leg.symbol)
            except Exception:
                ltp = None
            price = round(ltp * amo_buffer, 2) if ltp else 0.01  # minimal safe fallback

        leg_doc = {
            "legId": leg_id,
            "groupId": group_id,
            "exchange": leg.exchange,
            "symbol": leg.symbol,
            "qty": leg.qty,
            "product": leg.product,
            "priceType": leg.priceType,
            "variety": variety,
            "orderType": order_type,
            "limitPrice": price,
            "status": "pending"
        }
        db.collection("order_legs").document(leg_id).set(leg_doc)

        try:
            order_id = adapter.place_order(
                uid,
                exchange=leg.exchange,
                symbol=leg.symbol,
                qty=leg.qty,
                product=leg.product,
                order_type=order_type,
                variety=variety,
                price=price,
                tag="WMGR-BUCKET"
            )

            hist = adapter.order_history(uid, order_id)
            filled = any(h.get("status") == "COMPLETE" for h in hist)
            avg_price = 0.0
            for h in hist[::-1]:
                if h.get("status") == "COMPLETE":
                    avg_price = h.get("average_price", 0.0)
                    break

            db.collection("order_legs").document(leg_id).update({
                "status": "complete" if filled else "placed",
                "brokerOrderId": order_id,
                "averagePrice": avg_price,
                "history": hist
            })
            leg_doc.update({
                "status": "complete" if filled else "placed",
                "brokerOrderId": order_id,
                "averagePrice": avg_price,
                "history": hist
            })
            legs_resp.append(leg_doc)

        except Exception as e:
            db.collection("order_legs").document(leg_id).update({"status": "failed", "error": str(e)})
            leg_doc.update({"status": "failed", "error": str(e)})
            legs_resp.append(leg_doc)
            errors = True

    db.collection("order_groups").document(group_id).update({"status": "failed" if errors else "complete"})
    return OrderGroupResponse(groupId=group_id, status="failed" if errors else "complete", legs=legs_resp)


# --- admin-only endpoint to set your own plan ---
@app.post("/dev/grant-plan")
def grant_plan(plan: str, ctx: Dict = Depends(get_user_ctx)):
    admin_email = os.getenv("ADMIN_EMAIL")
    if not admin_email or ctx.get("email") != admin_email:
        raise HTTPException(403, "Not allowed")
    fb_auth.set_custom_user_claims(ctx["uid"], {"plan": plan})
    return {"ok": True, "plan": plan}

PLAN_PRICING = {
    # These are amounts in paise → Razorpay sees them as 1000 / 3000 / 5000 INR
    "standard": 100000,   # ₹1,000.00  (Safety plan)
    "pro":      300000,   # ₹3,000.00  (Balanced plan)
    "max":      500000,   # ₹5,000.00  (Growth plan)
}

# You can fine-tune unlock limits later if you use them for feature-gating
UNLOCK_LIMIT = {"standard": 5, "pro": 8, "max": 10}


# ---------- BILLING ----------
@app.post("/billing/order")
def create_billing_order(body: Dict, ctx: Dict = Depends(get_user_ctx)):
    """
    Create a Razorpay order for a chosen plan. Returns { key, order } for Checkout.
    body: { "plan": "standard" | "pro" | "max" }
    """
    plan = (body or {}).get("plan", "")
    amount = PLAN_PRICING.get(plan)
    if not amount:
        raise HTTPException(400, "Unknown plan")

    uid = ctx["uid"]
    client_local = _rzp_client()

    receipt = f"sub_{uid[:8]}_{int(time.time())}"
    order = client_local.order.create({
        "amount": amount,
        "currency": BILLING_CCY,
        "receipt": receipt,
        "payment_capture": 1,    # auto-capture
        "notes": {"uid": uid, "plan": plan},
    })

    # index intent for later reconciliation
    db.collection("subscription_intents").document(order["id"]).set({
        "uid": uid,
        "plan": plan,
        "amount": amount,
        "currency": BILLING_CCY,
        "createdAt": firestore.SERVER_TIMESTAMP,
        "status": "created",
    })

    return {"key": RZP_KEY_ID, "order": order}

@app.post("/billing/confirm")
def confirm_billing_payment(body: Dict, ctx: Dict = Depends(get_user_ctx)):
    """
    Verify payment signature from frontend and activate subscription.
    body: { razorpay_order_id, razorpay_payment_id, razorpay_signature, plan }
    """
    required = ["razorpay_order_id", "razorpay_payment_id", "razorpay_signature", "plan"]
    if not all(k in (body or {}) for k in required):
        raise HTTPException(400, "Missing fields")

    order_id = body["razorpay_order_id"]
    payment_id = body["razorpay_payment_id"]
    signature = body["razorpay_signature"]
    plan = body["plan"]
    amount = PLAN_PRICING.get(plan)
    if not amount:
        raise HTTPException(400, "Bad plan")

    # Verify signature: HMAC_SHA256(order_id|payment_id, KEY_SECRET)
    msg = f"{order_id}|{payment_id}".encode()
    expected = hmac.new(RZP_KEY_SECRET.encode(), msg, hashlib.sha256).hexdigest()
    if not hmac.compare_digest(expected, signature):
        raise HTTPException(400, "Invalid signature")

    uid = ctx["uid"]

    # Mark intent paid (best-effort)
    db.collection("subscription_intents").document(order_id).set({
        "uid": uid, "plan": plan, "amount": amount, "currency": BILLING_CCY,
        "status": "paid", "paymentId": payment_id, "updatedAt": firestore.SERVER_TIMESTAMP
    }, merge=True)

    # Create/replace current subscription (simple 30-day period)
    period_start = _now_ist()
    period_end = period_start + timedelta(days=30)

    sub_doc = {
        "uid": uid,
        "plan": plan,
        "status": "active",
        "currentPeriodStart": period_start.isoformat(),
        "currentPeriodEnd": period_end.isoformat(),
        "orderId": order_id,
        "paymentId": payment_id,
        "amount": amount,
        "currency": BILLING_CCY,
        "createdAt": firestore.SERVER_TIMESTAMP,
    }
    # store under user for quick user-centric views
    db.collection("users").document(uid).collection("subscriptions").document(order_id).set(sub_doc)
    # and global index for admin metrics
    db.collection("subscriptions_index").document(order_id).set(sub_doc)

    # Grant entitlement via custom claim
    fb_auth.set_custom_user_claims(uid, {"plan": plan})

    return {"ok": True, "plan": plan, "periodEnd": period_end.isoformat()}

# Optional: production webhook (ngrok + Dashboard -> Webhooks)
@app.post("/billing/webhook")
async def razorpay_webhook(request: Request):
    if not RZP_WEBHOOK_SECRET:
        raise HTTPException(501, "Webhook secret not configured")
    body = await request.body()
    sig = request.headers.get("X-Razorpay-Signature", "")

    expected = hmac.new(RZP_WEBHOOK_SECRET.encode(), body, hashlib.sha256).hexdigest()
    if not hmac.compare_digest(expected, sig):
        raise HTTPException(400, "Bad signature")

    payload = await request.json()
    # Handle events like payment.captured or subscription.activated here if you switch to RZP Subscriptions product
    return {"ok": True}


# ---- admin console endpoints ----
@app.get("/admin/metrics")
def admin_metrics(_: Dict = Depends(require_admin)):


    def count(plan):
        return len(list(
            db.collection("subscriptions_index")
              .where("plan","==",plan)
              .where("status","==","active").stream()
        ))

    active_counts = {
        "standard": count("standard"),
        "pro": count("pro"),
        "max": count("max"),
    }
    # Lightweight summary expected by frontend
    buckets = load_buckets()
    buckets_count = len(buckets.get("buckets", [])) if isinstance(buckets, dict) else 0
    try:
        payments_count = sum(1 for _ in db.collection("subscription_intents").stream())
    except Exception:
        payments_count = 0

    out = {
        "bucketsCount": buckets_count,
        "activeSubsCount": sum(active_counts.values()),
        "paymentsCount": payments_count,
        "active": active_counts,
        "last10": []
    }
    for doc in db.collection("subscriptions_index").order_by(
        "createdAt", direction=firestore.Query.DESCENDING
    ).limit(10).stream():
        d = doc.to_dict()
        out["last10"].append({
            "uid": d.get("uid"),
            "plan": d.get("plan"),
            "status": d.get("status"),
            "amount": d.get("amount"),
            "when": d.get("createdAt", None).__class__.__name__ == "Timestamp"
                    and d["createdAt"].isoformat() or "",
        })
    return out

@app.get("/admin/stats")
def admin_stats(_: Dict = Depends(require_admin)):
    # lightweight counts for now
    users_count = sum(1 for _ in db.collection("users").stream())
    subs_count = sum(1 for _ in db.collection("subscriptions").stream())
    orders_count = sum(1 for _ in db.collection("order_groups").stream())
    return {"users": users_count, "subscriptions": subs_count, "orderGroups": orders_count}
