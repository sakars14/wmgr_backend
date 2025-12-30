from __future__ import annotations

import math
import os
import uuid
from typing import Any, Dict, List, Optional
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from google.cloud import firestore

from adapters.zerodha_adapter import ZerodhaAdapter
from app.deps.authz import get_uid, require_admin
from market_time import choose_order_mode


router = APIRouter(prefix="/model-buckets", tags=["model-buckets"])

db = firestore.Client()

RISK_BANDS = ["Conservative", "Balanced", "Aggressive"]
RISK_BAND_ORDER = {name: idx for idx, name in enumerate(RISK_BANDS)}
WEIGHT_EPSILON = 0.01


class OrderPreviewRequest(BaseModel):
    budgetInr: float


class HoldingIn(BaseModel):
    exchange: str
    symbol: str
    name: str
    assetClass: str
    weight: float


class CreateVersionIn(BaseModel):
    versionId: Optional[str] = None
    holdings: List[HoldingIn]
    notes: Optional[str] = None


class UpdateBucketIn(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    riskBand: Optional[str] = None
    assetMix: Optional[Dict[str, float]] = None
    isActive: Optional[bool] = None


def _normalize_risk_band(raw: Optional[str]) -> str:
    if not raw:
        return "Balanced"
    text = str(raw).strip().lower()
    if "conservative" in text or text in {"low", "safe"}:
        return "Conservative"
    if "aggressive" in text or text in {"high"}:
        return "Aggressive"
    if "balanced" in text or "moderate" in text or text in {"medium"}:
        return "Balanced"
    return "Balanced"


def _sum_weights(values: List[float]) -> float:
    total = 0.0
    for v in values:
        total += float(v or 0.0)
    return total


def _validate_asset_mix(asset_mix: Dict[str, Any]) -> None:
    weights = [
        asset_mix.get("equity"),
        asset_mix.get("debt"),
        asset_mix.get("gold"),
        asset_mix.get("cash"),
    ]
    if any(v is None for v in weights):
        raise HTTPException(status_code=400, detail="assetMix must include equity/debt/gold/cash")
    if any(float(v) < 0 for v in weights):
        raise HTTPException(status_code=400, detail="assetMix weights must be non-negative")
    total = _sum_weights(weights)
    if abs(total - 1.0) > WEIGHT_EPSILON:
        raise HTTPException(status_code=400, detail="assetMix weights must sum to 1.0")


def _validate_holdings(holdings: List[Dict[str, Any]]) -> None:
    if not holdings:
        raise HTTPException(status_code=400, detail="Holdings required")
    weights = []
    for h in holdings:
        weight = float(h.get("weight") or 0.0)
        if weight <= 0:
            raise HTTPException(status_code=400, detail="Holdings weights must be positive")
        weights.append(weight)
    total = _sum_weights(weights)
    if abs(total - 1.0) > WEIGHT_EPSILON:
        raise HTTPException(status_code=400, detail="Holdings weights must sum to 1.0")


def _get_bucket_and_version(bucket_id: str) -> tuple[Dict[str, Any], Dict[str, Any], List[Dict[str, Any]]]:
    ref = db.collection("modelBuckets").document(bucket_id)
    snap = ref.get()
    if not snap.exists:
        raise HTTPException(status_code=404, detail="Bucket not found")

    bucket = snap.to_dict() or {}
    bucket["bucketId"] = bucket.get("bucketId") or snap.id

    version_id = bucket.get("publishedVersionId")
    if not version_id:
        raise HTTPException(status_code=404, detail="Published version missing")

    version_snap = ref.collection("versions").document(version_id).get()
    if not version_snap.exists:
        raise HTTPException(status_code=404, detail="Published version not found")

    version = version_snap.to_dict() or {}
    version["versionId"] = version.get("versionId") or version_id
    holdings = version.get("holdings") or []

    return bucket, version, holdings


def _price_holdings(uid: str, holdings: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    adapter = ZerodhaAdapter(db)
    priced = []
    for h in holdings:
        exchange = h.get("exchange") or "NSE"
        symbol = h.get("symbol")
        price = None
        if symbol:
            try:
                price = adapter.get_ltp(uid, exchange, symbol)
            except Exception:
                price = None
        priced.append(
            {
                **h,
                "exchange": exchange,
                "symbol": symbol,
                "ltp": price,
                "price": price,
            }
        )
    return priced


def _build_order_preview(budget_inr: float, priced_holdings: List[Dict[str, Any]]) -> Dict[str, Any]:
    if budget_inr <= 0:
        raise HTTPException(status_code=400, detail="budgetInr must be > 0")

    _validate_holdings(priced_holdings)

    priced = [h for h in priced_holdings if isinstance(h.get("price"), (int, float)) and h["price"] > 0]
    if not priced:
        raise HTTPException(status_code=400, detail="Connect Zerodha to fetch live prices")

    working = []
    for h in priced_holdings:
        price = h.get("price")
        weight = float(h.get("weight") or 0.0)
        qty = 0
        line_total = 0.0
        if isinstance(price, (int, float)) and price > 0 and weight > 0:
            target_amount = budget_inr * weight
            qty = int(math.floor(target_amount / price))
            line_total = qty * price
        working.append(
            {
                "exchange": h.get("exchange") or "NSE",
                "symbol": h.get("symbol"),
                "weight": weight,
                "price": price if isinstance(price, (int, float)) and price > 0 else None,
                "qty": qty,
                "lineTotal": line_total,
            }
        )

    subtotal = sum(item["lineTotal"] for item in working)
    leftover = budget_inr - subtotal

    warnings: List[str] = []
    eligible = [h for h in working if h["price"] is not None]
    if eligible:
        eligible.sort(key=lambda item: item["weight"], reverse=True)
        min_price = min(item["price"] for item in eligible if item["price"] is not None)
        max_iterations = 10000
        iterations = 0
        while leftover >= min_price and iterations < max_iterations:
            placed = False
            for item in eligible:
                if item["price"] <= leftover:
                    item["qty"] += 1
                    item["lineTotal"] += item["price"]
                    leftover -= item["price"]
                    placed = True
                    break
            if not placed:
                break
            iterations += 1
        if iterations >= max_iterations:
            warnings.append("Leftover allocation stopped early due to safety limit.")

    filtered = [h for h in working if h["qty"] > 0]
    if len(filtered) < len(working):
        warnings.append("Some holdings dropped due to zero quantity or missing prices.")

    subtotal = sum(item["lineTotal"] for item in filtered)
    leftover = budget_inr - subtotal

    return {
        "budgetInr": budget_inr,
        "holdings": filtered,
        "subtotal": subtotal,
        "leftover": leftover,
        "warnings": warnings,
    }


DEFAULT_BUCKETS = [
    {
        "bucketId": "conservative_v1",
        "name": "Conservative Core",
        "riskBand": "Conservative",
        "description": "Debt-heavy ETF mix with a smaller equity and gold tilt for stability.",
        "assetMix": {"equity": 0.25, "debt": 0.65, "gold": 0.10, "cash": 0.0},
        "holdings": [
            {"exchange": "NSE", "symbol": "NIFTYBEES", "name": "Nippon India ETF Nifty BeES", "assetClass": "equity", "weight": 0.15},
            {"exchange": "NSE", "symbol": "JUNIORBEES", "name": "Nippon India ETF Junior BeES", "assetClass": "equity", "weight": 0.10},
            {"exchange": "NSE", "symbol": "LIQUIDBEES", "name": "Nippon India ETF Liquid BeES", "assetClass": "debt", "weight": 0.65},
            {"exchange": "NSE", "symbol": "GOLDBEES", "name": "Nippon India ETF Gold BeES", "assetClass": "gold", "weight": 0.10},
        ],
    },
    {
        "bucketId": "balanced_v1",
        "name": "Balanced Core",
        "riskBand": "Balanced",
        "description": "Balanced ETF mix with equity growth and a stabilizing debt sleeve.",
        "assetMix": {"equity": 0.55, "debt": 0.35, "gold": 0.10, "cash": 0.0},
        "holdings": [
            {"exchange": "NSE", "symbol": "NIFTYBEES", "name": "Nippon India ETF Nifty BeES", "assetClass": "equity", "weight": 0.35},
            {"exchange": "NSE", "symbol": "JUNIORBEES", "name": "Nippon India ETF Junior BeES", "assetClass": "equity", "weight": 0.20},
            {"exchange": "NSE", "symbol": "LIQUIDBEES", "name": "Nippon India ETF Liquid BeES", "assetClass": "debt", "weight": 0.35},
            {"exchange": "NSE", "symbol": "GOLDBEES", "name": "Nippon India ETF Gold BeES", "assetClass": "gold", "weight": 0.10},
        ],
    },
    {
        "bucketId": "aggressive_v1",
        "name": "Aggressive Core",
        "riskBand": "Aggressive",
        "description": "Equity-heavy ETF mix designed for long-term growth with minimal defensive tilt.",
        "assetMix": {"equity": 0.75, "debt": 0.20, "gold": 0.05, "cash": 0.0},
        "holdings": [
            {"exchange": "NSE", "symbol": "NIFTYBEES", "name": "Nippon India ETF Nifty BeES", "assetClass": "equity", "weight": 0.45},
            {"exchange": "NSE", "symbol": "JUNIORBEES", "name": "Nippon India ETF Junior BeES", "assetClass": "equity", "weight": 0.30},
            {"exchange": "NSE", "symbol": "LIQUIDBEES", "name": "Nippon India ETF Liquid BeES", "assetClass": "debt", "weight": 0.20},
            {"exchange": "NSE", "symbol": "GOLDBEES", "name": "Nippon India ETF Gold BeES", "assetClass": "gold", "weight": 0.05},
        ],
    },
]


@router.post("/admin/seed")
def seed_model_buckets(admin_ctx: Dict[str, Any] = Depends(require_admin)):
    created: List[str] = []
    for entry in DEFAULT_BUCKETS:
        bucket_id = entry["bucketId"]
        ref = db.collection("modelBuckets").document(bucket_id)
        if ref.get().exists:
            continue

        asset_mix = entry["assetMix"]
        holdings = entry["holdings"]

        _validate_asset_mix(asset_mix)
        _validate_holdings(holdings)

        header = {
            "bucketId": bucket_id,
            "name": entry["name"],
            "riskBand": entry["riskBand"],
            "description": entry["description"],
            "assetMix": asset_mix,
            "publishedVersionId": "v1",
            "isActive": True,
            "updatedAt": firestore.SERVER_TIMESTAMP,
            "createdAt": firestore.SERVER_TIMESTAMP,
        }
        ref.set(header)

        version_doc = {
            "versionId": "v1",
            "holdings": holdings,
            "notes": "Seeded default model bucket.",
            "createdAt": firestore.SERVER_TIMESTAMP,
        }
        ref.collection("versions").document("v1").set(version_doc)
        created.append(bucket_id)

    return {"ok": True, "seeded": len(created), "created": created}


@router.get("/admin/all")
def admin_model_buckets(admin_ctx: Dict[str, Any] = Depends(require_admin)):
    buckets: List[Dict[str, Any]] = []
    for snap in db.collection("modelBuckets").stream():
        data = snap.to_dict() or {}
        bucket_id = data.get("bucketId") or snap.id
        versions: Dict[str, Any] = {}
        for version_snap in snap.reference.collection("versions").stream():
            version_data = version_snap.to_dict() or {}
            version_id = version_data.get("versionId") or version_snap.id
            versions[version_id] = {
                "versionId": version_id,
                "holdings": version_data.get("holdings") or [],
                "notes": version_data.get("notes"),
                "createdAt": version_data.get("createdAt"),
                "updatedAt": version_data.get("updatedAt"),
            }
        buckets.append(
            {
                "bucketId": bucket_id,
                "name": data.get("name"),
                "description": data.get("description"),
                "riskBand": data.get("riskBand"),
                "assetMix": data.get("assetMix"),
                "isActive": data.get("isActive"),
                "publishedVersionId": data.get("publishedVersionId"),
                "createdAt": data.get("createdAt"),
                "updatedAt": data.get("updatedAt"),
                "versions": versions,
            }
        )

    return {"buckets": buckets}


@router.patch("/admin/{bucket_id}")
def admin_update_bucket(
    bucket_id: str,
    payload: UpdateBucketIn,
    admin_ctx: Dict[str, Any] = Depends(require_admin),
):
    updates: Dict[str, Any] = {}
    if payload.name is not None:
        updates["name"] = payload.name
    if payload.description is not None:
        updates["description"] = payload.description
    if payload.riskBand is not None:
        if payload.riskBand not in RISK_BANDS:
            raise HTTPException(status_code=400, detail="Invalid riskBand")
        updates["riskBand"] = payload.riskBand
    if payload.assetMix is not None:
        _validate_asset_mix(payload.assetMix)
        updates["assetMix"] = payload.assetMix
    if payload.isActive is not None:
        updates["isActive"] = payload.isActive

    if not updates:
        raise HTTPException(status_code=400, detail="No updates provided")

    updates["updatedAt"] = firestore.SERVER_TIMESTAMP
    ref = db.collection("modelBuckets").document(bucket_id)
    if not ref.get().exists:
        raise HTTPException(status_code=404, detail="Bucket not found")
    ref.set(updates, merge=True)

    return {"ok": True, "bucketId": bucket_id, "updated": list(updates.keys())}


@router.post("/admin/{bucket_id}/versions")
def admin_create_version(
    bucket_id: str,
    payload: CreateVersionIn,
    admin_ctx: Dict[str, Any] = Depends(require_admin),
):
    ref = db.collection("modelBuckets").document(bucket_id)
    if not ref.get().exists:
        raise HTTPException(status_code=404, detail="Bucket not found")

    version_id = payload.versionId
    if not version_id:
        version_id = f"v{datetime.utcnow().strftime('%Y%m%d%H%M%S')}"

    version_ref = ref.collection("versions").document(version_id)
    if version_ref.get().exists:
        raise HTTPException(status_code=409, detail="Version already exists")

    holdings = [h.dict() for h in payload.holdings]
    _validate_holdings(holdings)

    version_ref.set(
        {
            "versionId": version_id,
            "holdings": holdings,
            "notes": payload.notes,
            "createdAt": firestore.SERVER_TIMESTAMP,
            "updatedAt": firestore.SERVER_TIMESTAMP,
        }
    )
    ref.set({"updatedAt": firestore.SERVER_TIMESTAMP}, merge=True)

    return {"ok": True, "bucketId": bucket_id, "versionId": version_id}


@router.post("/admin/{bucket_id}/publish/{version_id}")
def admin_publish_version(
    bucket_id: str,
    version_id: str,
    admin_ctx: Dict[str, Any] = Depends(require_admin),
):
    ref = db.collection("modelBuckets").document(bucket_id)
    if not ref.get().exists:
        raise HTTPException(status_code=404, detail="Bucket not found")

    version_ref = ref.collection("versions").document(version_id)
    if not version_ref.get().exists:
        raise HTTPException(status_code=404, detail="Version not found")

    ref.set(
        {
            "publishedVersionId": version_id,
            "updatedAt": firestore.SERVER_TIMESTAMP,
        },
        merge=True,
    )

    return {"ok": True, "bucketId": bucket_id, "publishedVersionId": version_id}


@router.get("/recommended")
def model_buckets_recommended(uid: str = Depends(get_uid)):

    plan_snap = db.collection("clientPlans").document(uid).get()
    plan_data = plan_snap.to_dict() if plan_snap.exists else {}

    risk_raw = None
    if isinstance(plan_data, dict):
        risk_raw = (plan_data.get("risk") or {}).get("finalBand")
        if not risk_raw:
            risk_raw = ((plan_data.get("plan") or {}).get("risk") or {}).get("finalBand")
        if not risk_raw:
            risk_raw = ((plan_data.get("plan") or {}).get("risk") or {}).get("quizLabel")
        if not risk_raw:
            risk_raw = ((plan_data.get("blueprint") or {}).get("derived") or {}).get("risk", {}).get("riskQuizLabel")

    if not risk_raw:
        profile_snap = db.collection("clientProfiles").document(uid).get()
        if profile_snap.exists:
            risk_raw = (profile_snap.to_dict() or {}).get("riskQuiz", {}).get("riskLabel")

    recommended_band = _normalize_risk_band(risk_raw)

    buckets: List[Dict[str, Any]] = []
    for snap in db.collection("modelBuckets").where("isActive", "==", True).stream():
        data = snap.to_dict() or {}
        risk_band = data.get("riskBand")
        if risk_band not in RISK_BANDS:
            continue
        version_id = data.get("publishedVersionId")
        holdings_count = 0
        if version_id:
            version_snap = snap.reference.collection("versions").document(version_id).get()
            if version_snap.exists:
                holdings_count = len((version_snap.to_dict() or {}).get("holdings") or [])

        buckets.append(
            {
                "bucketId": data.get("bucketId") or snap.id,
                "name": data.get("name"),
                "riskBand": risk_band,
                "description": data.get("description"),
                "assetMix": data.get("assetMix"),
                "publishedVersionId": version_id,
                "holdingsCount": holdings_count,
            }
        )

    buckets.sort(key=lambda b: RISK_BAND_ORDER.get(b["riskBand"], 999))
    return {"recommendedRiskBand": recommended_band, "buckets": buckets}


@router.get("/{bucket_id}")
def model_bucket_detail(bucket_id: str, uid: str = Depends(get_uid)):

    bucket, version, holdings = _get_bucket_and_version(bucket_id)
    priced_holdings_raw = _price_holdings(uid, holdings)
    priced_holdings = [
        {
            "exchange": h.get("exchange") or "NSE",
            "symbol": h.get("symbol"),
            "assetClass": h.get("assetClass"),
            "weight": h.get("weight"),
            "ltp": h.get("ltp"),
            "price": h.get("price"),
        }
        for h in priced_holdings_raw
    ]

    return {
        "bucket": bucket,
        "version": {"versionId": version.get("versionId"), "holdings": holdings},
        "pricedHoldings": priced_holdings,
    }


@router.post("/{bucket_id}/order-preview")
def model_bucket_order_preview(
    bucket_id: str,
    req: OrderPreviewRequest,
    uid: str = Depends(get_uid),
):

    _, _, holdings = _get_bucket_and_version(bucket_id)
    priced_holdings = _price_holdings(uid, holdings)
    return _build_order_preview(req.budgetInr, priced_holdings)


@router.post("/{bucket_id}/buy")
def model_bucket_buy(
    bucket_id: str,
    req: OrderPreviewRequest,
    uid: str = Depends(get_uid),
):

    _, _, holdings = _get_bucket_and_version(bucket_id)
    priced_holdings = _price_holdings(uid, holdings)
    preview = _build_order_preview(req.budgetInr, priced_holdings)

    if not preview["holdings"]:
        raise HTTPException(status_code=400, detail="Budget too low to place any orders")

    group_id = uuid.uuid4().hex
    db.collection("order_groups").document(group_id).set(
        {
            "groupId": group_id,
            "uid": uid,
            "bucketId": bucket_id,
            "status": "pending",
            "createdAt": firestore.SERVER_TIMESTAMP,
        }
    )

    adapter = ZerodhaAdapter(db)
    legs_resp: List[Dict[str, Any]] = []
    errors = False
    amo_buffer = float(os.getenv("AMO_LIMIT_BUFFER", "1.01"))

    for item in preview["holdings"]:
        leg_id = uuid.uuid4().hex
        exchange = item.get("exchange") or "NSE"
        symbol = item.get("symbol")
        qty = int(item.get("qty") or 0)
        price = item.get("price") if isinstance(item.get("price"), (int, float)) else None
        if qty <= 0 or not symbol:
            continue

        variety, order_type = choose_order_mode("MARKET")
        limit_price = None
        if order_type == "LIMIT":
            if price:
                limit_price = round(price * amo_buffer, 2)
            else:
                limit_price = 0.01

        leg_doc = {
            "legId": leg_id,
            "groupId": group_id,
            "exchange": exchange,
            "symbol": symbol,
            "qty": qty,
            "product": "CNC",
            "priceType": "MARKET",
            "variety": variety,
            "orderType": order_type,
            "limitPrice": limit_price,
            "status": "pending",
        }
        db.collection("order_legs").document(leg_id).set(leg_doc)

        try:
            order_id = adapter.place_order(
                uid,
                exchange=exchange,
                symbol=symbol,
                qty=qty,
                product="CNC",
                order_type=order_type,
                variety=variety,
                price=limit_price,
                tag="WMGR-MODEL-BUCKET",
            )

            hist = adapter.order_history(uid, order_id)
            filled = any(h.get("status") == "COMPLETE" for h in hist)
            avg_price = 0.0
            for h in hist[::-1]:
                if h.get("status") == "COMPLETE":
                    avg_price = h.get("average_price", 0.0)
                    break

            db.collection("order_legs").document(leg_id).update(
                {
                    "status": "complete" if filled else "placed",
                    "brokerOrderId": order_id,
                    "averagePrice": avg_price,
                    "history": hist,
                }
            )
            leg_doc.update(
                {
                    "status": "complete" if filled else "placed",
                    "brokerOrderId": order_id,
                    "averagePrice": avg_price,
                    "history": hist,
                }
            )
            legs_resp.append(leg_doc)
        except Exception as e:
            db.collection("order_legs").document(leg_id).update(
                {"status": "failed", "error": str(e)}
            )
            leg_doc.update({"status": "failed", "error": str(e)})
            legs_resp.append(leg_doc)
            errors = True

    db.collection("order_groups").document(group_id).update(
        {"status": "failed" if errors else "complete"}
    )

    return {
        "ok": True,
        "preview": preview,
        "orderGroupResponse": {
            "groupId": group_id,
            "status": "failed" if errors else "complete",
            "legs": legs_resp,
        },
    }
