# backend/app/routers/plans.py
from __future__ import annotations

import os
from copy import deepcopy
from datetime import datetime, timezone
from typing import Any, Dict, Optional, List, Literal
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

import firebase_admin
from firebase_admin import credentials, firestore

from app.deps.authz import get_user_ctx, is_admin, require_admin
from app.planner.engine import generate_plans
from app.planner.deterministic import build_financial_blueprint
from app.llm.narrator import generate_narration, NarrationResponse

router = APIRouter(tags=["plans"])


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
    Accepts decoded Firebase context; allows admin via shared authz rules.
    """
    allowed, _ = is_admin(ctx)
    if allowed:
        return

    uid = _get_uid(ctx)
    if not uid:
        raise HTTPException(status_code=401, detail="Unauthenticated")

    raise HTTPException(status_code=403, detail="Admin only")


EXPECTED_BLUEPRINT_VERSION = "phase1-blueprint-v1"
VALID_PLAN_VISIBILITIES = {"draft", "shared", "revoked"}
SIMULATION_VERSION = "phase3-sim-v1"

LIABILITY_OUTSTANDING_COMPONENT_KEYS = [
    "houseLoan1",
    "houseLoan2",
    "loanAgainstShares",
    "personalLoan1",
    "personalLoan2",
    "creditCard1",
    "creditCard2",
    "vehicleLoan",
    "others",
]

LIABILITY_EMI_KEYS = {
    "houseLoan1": "houseLoan1Emi",
    "houseLoan2": "houseLoan2Emi",
    "loanAgainstShares": "loanAgainstSharesEmi",
    "personalLoan1": "personalLoan1Emi",
    "personalLoan2": "personalLoan2Emi",
    "creditCard1": "creditCard1Emi",
    "creditCard2": "creditCard2Emi",
    "vehicleLoan": "vehicleLoanEmi",
    "others": "othersEmi",
}

AUTO_DEBT_PRIORITY = [
    "creditCard1",
    "creditCard2",
    "personalLoan1",
    "personalLoan2",
    "loanAgainstShares",
    "others",
    "vehicleLoan",
    "houseLoan1",
    "houseLoan2",
]


def _is_blueprint_compatible(blueprint: Optional[Dict[str, Any]]) -> bool:
    if not isinstance(blueprint, dict):
        return False
    meta = blueprint.get("meta") or {}
    if meta.get("blueprintVersion") != EXPECTED_BLUEPRINT_VERSION:
        return False
    emergency = (blueprint.get("derived") or {}).get("emergency") or {}
    return all(k in emergency for k in ["gap", "surplus", "status"])


def _normalize_plan_visibility(data: Optional[Dict[str, Any]]) -> str:
    if not isinstance(data, dict):
        return "draft"
    visibility = data.get("planVisibility")
    if visibility in VALID_PLAN_VISIBILITIES:
        return visibility
    if data.get("shared") is True:
        return "shared"
    if data.get("shared") is False:
        return "draft"
    return "draft"


def _visibility_payload(
    existing_data: Optional[Dict[str, Any]],
    visibility: Optional[str] = None,
    set_timestamp: bool = False,
) -> Dict[str, Any]:
    resolved = visibility or _normalize_plan_visibility(existing_data)
    payload: Dict[str, Any] = {
        "planVisibility": resolved,
        "shared": resolved == "shared",
    }
    if set_timestamp:
        payload["visibilityUpdatedAt"] = firestore.SERVER_TIMESTAMP
    return payload


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _safe_float(v: Any) -> float:
    try:
        if v is None:
            return 0.0
        if isinstance(v, (int, float)):
            return float(v)
        if isinstance(v, str) and v.strip() == "":
            return 0.0
        return float(v)
    except Exception:
        return 0.0


def _priority_rank(value: Any) -> int:
    if value is None:
        return 3
    p = str(value).strip().lower()
    if p.startswith("high"):
        return 0
    if p.startswith("medium"):
        return 1
    if p.startswith("low"):
        return 2
    return 3


def _build_goal_recommendations(blueprint: Dict[str, Any]) -> Dict[str, Any]:
    derived = blueprint.get("derived") or {}
    projections = derived.get("goalProjections") or []
    sorted_goals = []
    for g in projections:
        if not isinstance(g, dict):
            continue
        rank = _priority_rank(g.get("priority"))
        horizon = _safe_float(g.get("horizonYears"))
        sorted_goals.append((rank, horizon, g))
    sorted_goals.sort(key=lambda item: (item[0], item[1]))

    top_goals = []
    for _, _, g in sorted_goals[:3]:
        top_goals.append(
            {
                "goalKey": g.get("goalKey"),
                "label": g.get("label"),
                "status": g.get("status"),
                "requiredMonthlyInvestmentRounded": g.get("requiredMonthlyInvestmentRounded"),
                "shortfallMonthlyRounded": g.get("shortfallMonthlyRounded"),
            }
        )

    guidance: List[str] = []
    if not projections:
        guidance.append("Add goals to see projections and monthly investment targets.")
    else:
        high_goals = [g for _, _, g in sorted_goals if _priority_rank(g.get("priority")) == 0]
        if high_goals:
            any_shortfall = any(g.get("status") in {"Shortfall", "NotFeasible"} for g in high_goals)
            if any_shortfall:
                total_shortfall = 0.0
                for g in high_goals:
                    if g.get("status") in {"Shortfall", "NotFeasible"}:
                        total_shortfall += _safe_float(g.get("shortfallMonthlyRounded"))
                if total_shortfall > 0:
                    guidance.append(
                        f"High priority goals need about INR {total_shortfall:.0f}/month more; consider extending horizon or reducing target."
                    )
                else:
                    guidance.append(
                        "High priority goals show a shortfall; consider extending horizon or reducing target."
                    )
            else:
                guidance.append(
                    "High priority goals look on track if you allocate surplus consistently."
                )
        else:
            guidance.append("Set priorities for your goals to improve projections.")

    return {
        "goalRecommendationsVersion": "phase3.1-v1",
        "topGoals": top_goals,
        "guidance": guidance,
    }


def _apply_sip_delta(
    profile: Dict[str, Any],
    sip_delta: float,
    sip_target: str,
    highlights: List[str],
) -> None:
    contributions = profile.get("contributions")
    if not isinstance(contributions, dict):
        contributions = {}
        profile["contributions"] = contributions
    if sip_delta == 0:
        return

    if sip_target == "EQUITY_MF":
        targets = [("sipEquityMfMonthly", sip_delta)]
    elif sip_target == "DEBT_MF":
        targets = [("sipDebtMfMonthly", sip_delta)]
    elif sip_target == "SPLIT_50_50":
        half = sip_delta / 2.0
        targets = [("sipEquityMfMonthly", half), ("sipDebtMfMonthly", half)]
    else:
        highlights.append("Unknown SIP target; no SIP adjustment applied.")
        return

    for key, delta in targets:
        current = _safe_float(contributions.get(key))
        proposed = current + delta
        updated = max(0.0, proposed)
        contributions[key] = updated
        if proposed < 0:
            highlights.append(f"{key} floored at 0 after SIP change.")


def _apply_debt_prepay(
    profile: Dict[str, Any],
    debt_prepay_monthly: float,
    projection_months: int,
    debt_target: str,
    highlights: List[str],
) -> Dict[str, Any]:
    liabilities = profile.get("liabilities")
    if not isinstance(liabilities, dict):
        liabilities = {}
        profile["liabilities"] = liabilities
    total_prepay = max(0.0, debt_prepay_monthly) * projection_months

    if total_prepay <= 0:
        return {
            "targetKeys": [],
            "beforeTotal": 0.0,
            "afterTotal": 0.0,
            "appliedAmount": 0.0,
            "fullyPaidKeys": [],
        }

    if debt_target == "AUTO":
        target_keys = AUTO_DEBT_PRIORITY
    elif debt_target == "CREDIT_CARD":
        target_keys = ["creditCard1", "creditCard2"]
    elif debt_target == "PERSONAL_LOAN":
        target_keys = ["personalLoan1", "personalLoan2"]
    elif debt_target == "VEHICLE_LOAN":
        target_keys = ["vehicleLoan"]
    elif debt_target == "OTHERS":
        target_keys = ["others", "loanAgainstShares"]
    else:
        highlights.append("Unknown debt target; no debt prepay applied.")
        return {
            "targetKeys": [],
            "beforeTotal": 0.0,
            "afterTotal": 0.0,
            "appliedAmount": 0.0,
            "fullyPaidKeys": [],
        }

    before_total = sum(_safe_float(liabilities.get(k)) for k in target_keys)
    remaining = total_prepay
    applied = 0.0
    fully_paid_keys: List[str] = []
    applied_keys: List[str] = []

    for key in target_keys:
        if remaining <= 0:
            break
        outstanding = _safe_float(liabilities.get(key))
        if outstanding <= 0:
            continue
        reduction = min(outstanding, remaining)
        new_outstanding = outstanding - reduction
        liabilities[key] = new_outstanding
        applied += reduction
        remaining -= reduction
        applied_keys.append(key)
        if new_outstanding <= 0:
            fully_paid_keys.append(key)
            emi_key = LIABILITY_EMI_KEYS.get(key)
            if emi_key:
                emi_val = _safe_float(liabilities.get(emi_key))
                if emi_val > 0:
                    liabilities[emi_key] = 0.0
                    highlights.append(f"{key} fully prepaid; EMI set to 0.")

    after_total = sum(_safe_float(liabilities.get(k)) for k in target_keys)

    if before_total <= 0:
        highlights.append("No matching outstanding debt found for the selected target.")
    elif applied > 0:
        highlights.append(
            f"Debt prepay applied: {applied:.0f} over {projection_months} months."
        )
    else:
        highlights.append("Debt prepay amount did not reduce outstanding balances.")

    if remaining > 0 and applied > 0:
        highlights.append("Prepay amount exceeded target debt; remaining amount unused.")

    total_outstanding = sum(
        _safe_float(liabilities.get(k)) for k in LIABILITY_OUTSTANDING_COMPONENT_KEYS
    )
    liabilities["totalOutstanding"] = total_outstanding

    return {
        "targetKeys": applied_keys or target_keys,
        "beforeTotal": before_total,
        "afterTotal": after_total,
        "appliedAmount": applied,
        "fullyPaidKeys": fully_paid_keys,
    }


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


class GenerateRequest(BaseModel):
    # If provided and != caller uid, caller must be admin
    clientId: Optional[str] = None
    source: Optional[str] = None


class SimulationRequest(BaseModel):
    projectionMonths: int = 12
    sipDeltaMonthly: float = 0.0
    sipTarget: Literal["EQUITY_MF", "DEBT_MF", "SPLIT_50_50"] = "EQUITY_MF"
    debtPrepayMonthly: float = 0.0
    debtTarget: Literal[
        "CREDIT_CARD",
        "PERSONAL_LOAN",
        "VEHICLE_LOAN",
        "OTHERS",
        "AUTO",
    ] = "AUTO"
    emergencyTargetMonths: Optional[int] = None
    includeNarration: bool = False


class SaveSimulationRequest(BaseModel):
    name: str
    simulationRequest: Dict[str, Any]
    simulationResponse: Dict[str, Any]


@router.post("/plans/recommendations")
def plans_recommendations(
    req: RecommendationsRequest,
    user: Dict[str, Any] = Depends(get_user_ctx),
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
    goal_recommendations = _build_goal_recommendations(blueprint)
    plan_model = dict(plan_model)
    plan_model["goalRecommendations"] = goal_recommendations

    # 3) Persist it (shared defaults to existing shared if present)
    ref = db.collection("clientPlans").document(target_uid)
    existing = ref.get()
    existing_data = existing.to_dict() if existing.exists else {}
    plan_visibility = _normalize_plan_visibility(existing_data)
    visibility_payload = _visibility_payload(existing_data, plan_visibility)
    if not existing_data or "planVisibility" not in existing_data:
        visibility_payload["visibilityUpdatedAt"] = firestore.SERVER_TIMESTAMP

    client_name = ""
    personal = (profile.get("personal") or {})
    if isinstance(personal, dict):
        client_name = personal.get("name", "") or ""

    ref.set(
        {
            "clientId": target_uid,
            "clientName": client_name,
            **visibility_payload,
            "updatedAt": firestore.SERVER_TIMESTAMP,
            "generatedAt": firestore.SERVER_TIMESTAMP,
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
    user: Dict[str, Any] = Depends(get_user_ctx),
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
    plan_visibility = _normalize_plan_visibility(existing_data)
    visibility_payload = _visibility_payload(existing_data, plan_visibility)

    # 2) Reuse stored blueprint/plan if available and compatible
    blueprint = existing_data.get("blueprint")
    plan_model = existing_data.get("plan")
    generated_plan = False
    generated_blueprint = False

    if not _is_blueprint_compatible(blueprint):
        blueprint = build_financial_blueprint(profile)
        generated_blueprint = True
    if not plan_model:
        plan_model = generate_plans(profile)
        generated_plan = True

    plan_model = dict(plan_model)
    plan_model["goalRecommendations"] = _build_goal_recommendations(blueprint)

    # 3) Generate narration
    narration = generate_narration(blueprint, plan_model)

    client_name = ""
    personal = (profile.get("personal") or {})
    if isinstance(personal, dict):
        client_name = personal.get("name", "") or ""

    update_payload: Dict[str, Any] = {
        "clientId": target_uid,
        "clientName": client_name,
        **visibility_payload,
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
    user: Dict[str, Any] = Depends(get_user_ctx),
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
    plan_visibility = _normalize_plan_visibility(existing_data)
    visibility_payload = _visibility_payload(existing_data, plan_visibility)
    if not existing_data or "planVisibility" not in existing_data:
        visibility_payload["visibilityUpdatedAt"] = firestore.SERVER_TIMESTAMP

    client_name = ""
    personal = (profile.get("personal") or {})
    if isinstance(personal, dict):
        client_name = personal.get("name", "") or ""

    ref.set(
        {
            "clientId": target_uid,
            "clientName": client_name,
            **visibility_payload,
            "updatedAt": firestore.SERVER_TIMESTAMP,
            "blueprint": blueprint,
            "blueprintVersion": blueprint.get("meta", {}).get("blueprintVersion", "phase1-blueprint-v1"),
        },
        merge=True,
    )

    return blueprint


# -------------------------
# Simulation endpoints
# -------------------------

@router.post("/plans/simulate")
def plans_simulate(
    req: SimulationRequest,
    user: Dict[str, Any] = Depends(get_user_ctx),
):
    db = _ensure_firebase_db()

    caller_uid = _get_uid(user)
    if not caller_uid:
        raise HTTPException(status_code=401, detail="Unauthenticated")

    snap = db.collection("clientProfiles").document(caller_uid).get()
    if not snap.exists:
        raise HTTPException(status_code=404, detail=f"clientProfiles/{caller_uid} not found")
    profile = snap.to_dict()

    projection_months = req.projectionMonths if req.projectionMonths and req.projectionMonths > 0 else 12

    base_blueprint = build_financial_blueprint(profile)

    simulated_profile = deepcopy(profile)
    highlights: List[str] = []

    if req.projectionMonths is not None and req.projectionMonths <= 0:
        highlights.append("Projection months invalid; defaulted to 12.")

    _apply_sip_delta(simulated_profile, req.sipDeltaMonthly, req.sipTarget, highlights)

    if req.emergencyTargetMonths is not None:
        emergency = simulated_profile.get("emergency")
        if not isinstance(emergency, dict):
            emergency = {}
            simulated_profile["emergency"] = emergency
        emergency["monthsTarget"] = req.emergencyTargetMonths

    debt_info = _apply_debt_prepay(
        simulated_profile,
        req.debtPrepayMonthly,
        projection_months,
        req.debtTarget,
        highlights,
    )

    simulated_blueprint = build_financial_blueprint(simulated_profile)
    simulated_blueprint["meta"] = simulated_blueprint.get("meta") or {}
    simulated_blueprint["meta"]["isScenario"] = True
    simulated_blueprint["meta"]["projectionMonths"] = projection_months
    simulated_blueprint["meta"]["simulationInputs"] = req.dict()

    base_surplus = _safe_float(((base_blueprint.get("derived") or {}).get("surplus") or {}).get("monthlySurplus"))
    sim_surplus = _safe_float(((simulated_blueprint.get("derived") or {}).get("surplus") or {}).get("monthlySurplus"))
    base_emergency = (base_blueprint.get("derived") or {}).get("emergency") or {}
    sim_emergency = (simulated_blueprint.get("derived") or {}).get("emergency") or {}
    base_gap = _safe_float(base_emergency.get("gap"))
    sim_gap = _safe_float(sim_emergency.get("gap"))

    deltas = {
        "monthlySurplusDelta": sim_surplus - base_surplus,
        "emergencyGapDelta": sim_gap - base_gap,
        "debtOutstandingBefore": debt_info.get("beforeTotal", 0.0),
        "debtOutstandingAfter": debt_info.get("afterTotal", 0.0),
        "debtOutstandingDelta": debt_info.get("afterTotal", 0.0) - debt_info.get("beforeTotal", 0.0),
        "debtTarget": req.debtTarget,
        "debtTargetKeys": debt_info.get("targetKeys", []),
    }

    scenario_summary = {
        "monthlySurplusBase": base_surplus,
        "monthlySurplusSimulated": sim_surplus,
        "emergencyGapBase": base_gap,
        "emergencyGapSimulated": sim_gap,
        "emergencyStatusBase": base_emergency.get("status"),
        "emergencyStatusSimulated": sim_emergency.get("status"),
        "debtOutstandingBefore": debt_info.get("beforeTotal", 0.0),
        "debtOutstandingAfter": debt_info.get("afterTotal", 0.0),
        "projectionMonths": projection_months,
    }

    narration_payload = None
    if req.includeNarration:
        plan_model = generate_plans(profile)
        narration_payload = generate_narration(simulated_blueprint, plan_model)

    response = {
        "simulationVersion": SIMULATION_VERSION,
        "generatedAt": _now_iso(),
        "baseBlueprint": base_blueprint,
        "simulatedBlueprint": simulated_blueprint,
        "scenarioSummary": scenario_summary,
        "deltas": deltas,
        "highlights": highlights,
        "narration": narration_payload.dict() if narration_payload else None,
    }
    return response


@router.post("/plans/simulations/save")
def plans_simulations_save(
    req: SaveSimulationRequest,
    user: Dict[str, Any] = Depends(get_user_ctx),
):
    db = _ensure_firebase_db()

    caller_uid = _get_uid(user)
    if not caller_uid:
        raise HTTPException(status_code=401, detail="Unauthenticated")

    name = (req.name or "").strip()
    if not name:
        raise HTTPException(status_code=400, detail="Simulation name is required")

    sim_id = uuid4().hex
    simulation_response = req.simulationResponse or {}
    deltas = simulation_response.get("deltas") or {}
    simulated_blueprint = simulation_response.get("simulatedBlueprint")
    highlights = simulation_response.get("highlights") or []
    simulation_version = simulation_response.get("simulationVersion") or SIMULATION_VERSION

    sim_doc = {
        "id": sim_id,
        "name": name,
        "createdAt": firestore.SERVER_TIMESTAMP,
        "simulationVersion": simulation_version,
        "simulationRequest": req.simulationRequest,
        "deltas": deltas,
        "highlights": highlights,
        "simulatedBlueprint": simulated_blueprint,
    }

    plan_ref = db.collection("clientPlans").document(caller_uid)
    plan_ref.collection("simulations").document(sim_id).set(sim_doc)
    plan_ref.set({"selectedSimulationId": sim_id}, merge=True)

    return {"ok": True, "id": sim_id}


# -------------------------
# Admin endpoints for UI
# -------------------------

@router.get("/admin/plans")
def admin_plans(admin_ctx: Dict[str, Any] = Depends(require_admin)):
    db = _ensure_firebase_db()

    # Best-effort ordering; if updatedAt isn't indexed yet, fallback to stream()
    plans_ref = db.collection("clientPlans")
    try:
        snaps = plans_ref.order_by("updatedAt", direction=firestore.Query.DESCENDING).limit(200).stream()
    except Exception:
        snaps = plans_ref.limit(200).stream()

    out: List[Dict[str, Any]] = []
    for s in snaps:
        d = s.to_dict() or {}
        plan_visibility = _normalize_plan_visibility(d)
        out.append(
            {
                "clientId": s.id,
                "clientName": d.get("clientName", ""),
                "planVisibility": plan_visibility,
                "updatedAt": d.get("updatedAt"),
            }
        )
    return out


@router.get("/admin/plans/{client_id}")
def admin_get_plan(client_id: str, admin_ctx: Dict[str, Any] = Depends(require_admin)):
    db = _ensure_firebase_db()

    snap = db.collection("clientPlans").document(client_id).get()
    if not snap.exists:
        raise HTTPException(status_code=404, detail=f"clientPlans/{client_id} not found")
    d = snap.to_dict() or {}
    return {
        "clientId": client_id,
        "clientName": d.get("clientName", ""),
        "planVisibility": _normalize_plan_visibility(d),
        "plan": d.get("plan"),
        "updatedAt": d.get("updatedAt"),
    }


class ShareRequest(BaseModel):
    shared: Optional[bool] = None
    visibility: Optional[str] = None


@router.post("/admin/plans/{client_id}/share")
def admin_share_plan(
    client_id: str,
    req: ShareRequest,
    admin_ctx: Dict[str, Any] = Depends(require_admin),
):
    db = _ensure_firebase_db()

    visibility = None
    if req.visibility in VALID_PLAN_VISIBILITIES:
        visibility = req.visibility
    elif req.shared is not None:
        visibility = "shared" if req.shared else "revoked"
    else:
        raise HTTPException(status_code=400, detail="Missing visibility")

    ref = db.collection("clientPlans").document(client_id)
    ref.set(
        {
            "planVisibility": visibility,
            "shared": visibility == "shared",
            "visibilityUpdatedAt": firestore.SERVER_TIMESTAMP,
            "updatedAt": firestore.SERVER_TIMESTAMP,
        },
        merge=True,
    )
    return {"ok": True, "clientId": client_id, "planVisibility": visibility}


# -------------------------
# User-facing plan visibility
# -------------------------

@router.get("/plans/shared/me")
def plans_shared_me(user: Dict[str, Any] = Depends(get_user_ctx)):
    db = _ensure_firebase_db()
    uid = _get_uid(user)
    if not uid:
        raise HTTPException(status_code=401, detail="Unauthenticated")

    snap = db.collection("clientPlans").document(uid).get()
    if not snap.exists:
        return {"hasPlan": False, "shared": False}

    data = snap.to_dict() or {}
    plan_visibility = _normalize_plan_visibility(data)
    return {
        "hasPlan": True,
        "shared": plan_visibility == "shared",
        "planVisibility": plan_visibility,
        "generatedAt": data.get("generatedAt"),
    }


@router.get("/plans/my")
def plans_my(user: Dict[str, Any] = Depends(get_user_ctx)):
    db = _ensure_firebase_db()
    uid = _get_uid(user)
    if not uid:
        raise HTTPException(status_code=401, detail="Unauthenticated")

    snap = db.collection("clientPlans").document(uid).get()
    if not snap.exists:
        raise HTTPException(status_code=404, detail="Plan not generated yet")

    data = snap.to_dict() or {}
    if _normalize_plan_visibility(data) != "shared":
        raise HTTPException(status_code=403, detail="Plan not shared yet")

    blueprint = data.get("blueprint")
    if not _is_blueprint_compatible(blueprint):
        profile_snap = db.collection("clientProfiles").document(uid).get()
        if profile_snap.exists:
            blueprint = build_financial_blueprint(profile_snap.to_dict())

    return {
        "plan": data.get("plan"),
        "blueprint": blueprint,
        "clientId": uid,
        "clientName": data.get("clientName", ""),
    }


@router.post("/plans/generate")
def plans_generate(
    req: GenerateRequest = GenerateRequest(),
    user: Dict[str, Any] = Depends(get_user_ctx),
):
    db = _ensure_firebase_db()

    caller_uid = _get_uid(user)
    if not caller_uid:
        raise HTTPException(status_code=401, detail="Unauthenticated")

    target_uid = caller_uid
    if req.clientId and req.clientId != caller_uid:
        _require_admin(db, user)
        target_uid = req.clientId

    snap = db.collection("clientProfiles").document(target_uid).get()
    if not snap.exists:
        raise HTTPException(status_code=404, detail=f"clientProfiles/{target_uid} not found")
    profile = snap.to_dict()

    blueprint = build_financial_blueprint(profile)
    plan_model = generate_plans(profile)
    goal_recommendations = _build_goal_recommendations(blueprint)
    plan_model = dict(plan_model)
    plan_model["goalRecommendations"] = goal_recommendations
    narration = generate_narration(blueprint, plan_model)
    warnings = blueprint.get("warnings") if isinstance(blueprint, dict) else None
    if not warnings:
        narration.clarifyingQuestions = []

    client_name = ""
    personal = (profile.get("personal") or {})
    if isinstance(personal, dict):
        client_name = personal.get("name", "") or ""

    source = req.source if req.source in {"onboarding-submit", "admin-manual", "profile-resubmit"} else "onboarding-submit"

    ref = db.collection("clientPlans").document(target_uid)
    ref.set(
        {
            "clientId": target_uid,
            "clientName": client_name,
            "plan": plan_model,
            "blueprint": blueprint,
            "blueprintVersion": blueprint.get("meta", {}).get("blueprintVersion", "phase1-blueprint-v1"),
            "llmNarration": narration.dict(),
            "narrationVersion": narration.narrationVersion,
            "narrationUpdatedAt": firestore.SERVER_TIMESTAMP,
            "generatedAt": firestore.SERVER_TIMESTAMP,
            "updatedAt": firestore.SERVER_TIMESTAMP,
            "planVisibility": "draft",
            "shared": False,
            "visibilityUpdatedAt": firestore.SERVER_TIMESTAMP,
            "lastGeneratedFrom": {"source": source},
        },
        merge=True,
    )

    return {"ok": True}
