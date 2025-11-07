# apps/api/main.py

from fastapi.responses import RedirectResponse
import os
import re
import uuid
import json
import time
import hmac, hashlib
import razorpay
from datetime import timedelta
from typing import Optional, Dict
from urllib.parse import quote
from datetime import datetime, time as dtime
from zoneinfo import ZoneInfo

from fastapi import FastAPI, Header, HTTPException, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

import firebase_admin
from firebase_admin import credentials, auth as fb_auth
from google.cloud import firestore

from adapters.zerodha_adapter import ZerodhaAdapter
from models import CreateOrderGroupRequest, OrderGroupResponse
from market_time import choose_order_mode


# ---------- Boot ----------
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"))

# FastAPI app (create ONCE)
app = FastAPI(title="WMGR API")

#Razorpay details

client = razorpay.Client(auth=(os.environ["RAZORPAY_KEY_ID"], os.environ["RAZORPAY_KEY_SECRET"]))

PLAN_PRICE_PAISE = {
    "standard": 19900,  # ₹199 → paise
    "pro":      49900,  # ₹499
    "max":      99900   # ₹999
}


# CORS: allow both localhost & 127.0.0.1 plus explicit APP_BASE_URL
_default_webs = ["http://127.0.0.1:3000", "http://localhost:3000"]
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


# ---------- Helpers ----------
def get_uid(authorization: Optional[str] = Header(None)) -> str:
    """Verify Firebase ID token from 'Authorization: Bearer <token>' and return uid."""
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(401, "Missing Bearer token")
    token = authorization.split(" ", 1)[1]
    try:
        decoded = fb_auth.verify_id_token(token)
    except Exception as e:
        # tiny clock skew tolerance
        if "Token used too early" in str(e):
            time.sleep(1)
            decoded = fb_auth.verify_id_token(token)
        else:
            raise HTTPException(401, f"Invalid token: {e}")
    return decoded["uid"]


def get_auth(authorization: Optional[str] = Header(None)) -> dict:
    """Return full decoded Firebase token (uid + email + custom claims)."""
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(401, "Missing Bearer token")
    token = authorization.split(" ", 1)[1]
    try:
        return fb_auth.verify_id_token(token)
    except Exception as e:
        if "Token used too early" in str(e):
            time.sleep(1)
            return fb_auth.verify_id_token(token)
        raise HTTPException(401, f"Invalid token: {e}")


def load_buckets():
    """Read demo buckets from seeds/buckets.json."""
    p = os.path.join(os.path.dirname(__file__), "seeds", "buckets.json")
    if not os.path.exists(p):
        return {"buckets": []}
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)["buckets"]


# helper that returns full decoded token (uid + email + claims)
def get_user_ctx(authorization: Optional[str] = Header(None)) -> Dict:
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(401, "Missing Bearer token")
    token = authorization.split(" ", 1)[1]
    try:
        decoded = fb_auth.verify_id_token(token)
        return decoded
    except Exception as e:
        if "Token used too early" in str(e):
            time.sleep(1)
            return fb_auth.verify_id_token(token)
        raise HTTPException(401, f"Invalid token: {e}")


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

def require_admin(ctx: Dict = Depends(get_user_ctx)):
    admin_email = (os.getenv("ADMIN_EMAIL","").strip().lower())
    is_admin = ctx.get("admin", False) or (ctx.get("email","").strip().lower() == admin_email)
    if not is_admin:
        raise HTTPException(403, "Not allowed")
    return ctx

# ---------- Probes ----------
@app.get("/")
def root():
    return {"ok": True, "service": "wmgr-api", "cors": allow_origins}


@app.get("/health")
def health():
    return {"ok": True}


@app.get("/dev/whoami")
def whoami(uid: str = Depends(get_uid)):
    u = fb_auth.get_user(uid)
    return {
        "uid": uid,
        "email": u.email,
        "claims": u.custom_claims or {},
        "admin_matches": ((u.email or "").strip().lower()
                          == (os.getenv("ADMIN_EMAIL", "").strip().lower()))
    }


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
    for b in load_buckets():
        if b["id"] == bucket_id:
            return b
    raise HTTPException(404, "Bucket not found")


# ---------- Zerodha OAuth ----------
@app.get("/auth/zerodha/login")
def zerodha_login(uid: str = Depends(get_uid)):
    from kiteconnect import KiteConnect

    api_key = os.getenv("ZERODHA_API_KEY")
    base_redirect = os.getenv("ZERODHA_REDIRECT_URL")
    if not api_key:
        raise HTTPException(500, "ZERODHA_API_KEY missing")
    if not base_redirect:
        raise HTTPException(500, "ZERODHA_REDIRECT_URL missing")

    k = KiteConnect(api_key=api_key)

    # Carry uid via redirect_url and also put it in state
    redirect_with_uid = f"{base_redirect}?uid={uid}"
    login_url = k.login_url() + "&redirect_url=" + quote(redirect_with_uid, safe="") + "&state=" + uid
    return {"loginUrl": login_url}


@app.get("/auth/zerodha/callback")
def zerodha_callback(request: Request):
    from kiteconnect import KiteConnect

    params = dict(request.query_params)
    request_token = params.get("request_token")
    uid = params.get("uid") or params.get("state")  # accept either

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

        db.collection("users").document(uid).collection("brokerLinks").document("zerodha").set(
            {"accessToken": access_token, "createdAt": firestore.SERVER_TIMESTAMP}
        )

        # 👇 Redirect the browser to your frontend buckets page
        app_base = os.getenv("APP_BASE_URL", "http://127.0.0.1:3000")
        return RedirectResponse(url=f"{app_base}/buckets?connected=1", status_code=302)

    except Exception as e:
        # if anything goes wrong, show structured error
        return {"ok": False, "stage": "exception", "error": str(e), "params": params}


# ---------- Orders (Plan-gated + AMO buffer) ----------

PLAN_LIMIT = {"none": 0, "standard": 5, "pro": 8, "max": 10}

def _derive_rank(bucket_id: str) -> int:
    """Extract a numeric order (1..N) from bucket id/name for gating."""
    m = re.search(r"(\d+)", bucket_id or "")
    return int(m.group(1)) if m else 999


@app.post("/orders/group", response_model=OrderGroupResponse)
def create_order_group(payload: CreateOrderGroupRequest, auth: dict = Depends(get_auth)):
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
    "standard": 19900,   # ₹199.00
    "pro":      49900,   # ₹499.00
    "max":      99900,   # ₹999.00
}
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
    client = _rzp_client()

    receipt = f"sub_{uid[:8]}_{int(time.time())}"
    order = client.order.create({
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


#----admin console endpoints ----
@app.get("/admin/metrics")
def admin_metrics(ctx: Dict = Depends(get_user_ctx)):
    admin_email = os.getenv("ADMIN_EMAIL", "")
    if not admin_email or ctx.get("email") != admin_email:
        raise HTTPException(403, "Not allowed")

    def count(plan):
        return len(list(
            db.collection("subscriptions_index")
              .where("plan","==",plan)
              .where("status","==","active").stream()
        ))

    out = {
        "active": {
            "standard": count("standard"),
            "pro": count("pro"),
            "max": count("max"),
        },
        "last10": []
    }
    for doc in db.collection("subscriptions_index").order_by("createdAt", direction=firestore.Query.DESCENDING).limit(10).stream():
        d = doc.to_dict()
        out["last10"].append({
            "uid": d.get("uid"),
            "plan": d.get("plan"),
            "status": d.get("status"),
            "amount": d.get("amount"),
            "when": d.get("createdAt", None).__class__.__name__ == "Timestamp" and d["createdAt"].isoformat() or "",
        })
    return out
@app.get("/admin/stats")
def admin_stats(_: Dict = Depends(require_admin)):
    # lightweight counts for now
    users_count = sum(1 for _ in db.collection("users").stream())
    subs_count = sum(1 for _ in db.collection("subscriptions").stream())
    orders_count = sum(1 for _ in db.collection("order_groups").stream())
    return {"users": users_count, "subscriptions": subs_count, "orderGroups": orders_count}