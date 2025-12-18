from typing import Dict, Any, List, Literal, Tuple
from dataclasses import dataclass
from .deterministic import build_financial_blueprint


RiskBand = Literal["Conservative", "Balanced", "Aggressive"]

@dataclass
class RiskCapacityResult:
    score: float
    age_score: float
    job_score: float
    buffer_score: float
    debt_score: float
    band: Literal["Low", "Medium", "High"]

@dataclass
class RiskResult:
    capacity: RiskCapacityResult
    quiz_score: float
    quiz_label: RiskBand | None
    final_band: RiskBand

@dataclass
class Allocation:
    equity: float
    debt: float
    gold: float
    cash: float
    real_estate_alt: float
    total: float

@dataclass
class PlanOutput:
    id: str
    name: str
    recommended: bool
    target_allocation_pct: Dict[str, float]
    target_allocation_value: Dict[str, float]
    current_allocation_pct: Dict[str, float]
    deltas_value: Dict[str, float]
    summary: str
    bullets: List[str]
    price: int


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


# ---------- Income, debt, buffer ----------

def compute_income_metrics(profile: Dict[str, Any]) -> Dict[str, float]:
    personal = profile.get("personal", {})
    cash_flow = profile.get("cashFlow", {})
    liabilities = profile.get("liabilities", {})

    monthly_income_in_hand = _safe_float(personal.get("monthlyIncomeInHand"))
    annual_bonus = _safe_float(personal.get("annualBonus"))
    passive_income = _safe_float(personal.get("passiveIncome"))
    other1 = _safe_float(personal.get("otherIncome1"))
    other2 = _safe_float(personal.get("otherIncome2"))

    annual_income = 12 * monthly_income_in_hand + annual_bonus + 12 * passive_income + 12 * other1 + 12 * other2
    monthly_income_total = annual_income / 12.0

    total_monthly_expense = _safe_float(cash_flow.get("totalMonthlyExpense"))

    # EMIs – adjust field names to match your Firestore doc
    emi_fields = [
        "homeLoan1Emi",
        "homeLoan2Emi",
        "personalLoan1Emi",
        "personalLoan2Emi",
        "vehicleLoanEmi",
        "loanAgainstSharesEmi",
        "creditCard1Emi",
        "creditCard2Emi",
        "otherLoanEmi",
    ]
    total_monthly_emi = sum(_safe_float(liabilities.get(f, 0)) for f in emi_fields)

    monthly_savings = monthly_income_total - total_monthly_expense - total_monthly_emi

    return {
        "annual_income": annual_income,
        "monthly_income_total": monthly_income_total,
        "total_monthly_expense": total_monthly_expense,
        "total_monthly_emi": total_monthly_emi,
        "monthly_savings": monthly_savings,
    }


def compute_liquid_assets(profile: Dict[str, Any]) -> Tuple[float, float]:
    assets = profile.get("assets", {})
    cash_flow = profile.get("cashFlow", {})

    bank1 = _safe_float(assets.get("bankBalance1"))
    bank2 = _safe_float(assets.get("bankBalance2"))
    fds = _safe_float(assets.get("fds"))
    rds = _safe_float(assets.get("rds"))
    post_office = _safe_float(assets.get("postOfficeSchemes"))
    debt_mf = _safe_float(assets.get("debtMf"))

    liquid_assets = bank1 + bank2 + fds + rds + post_office + debt_mf
    total_monthly_expense = _safe_float(cash_flow.get("totalMonthlyExpense"))

    if total_monthly_expense <= 0:
        months_of_runway = 0.0
    else:
        months_of_runway = liquid_assets / total_monthly_expense

    return liquid_assets, months_of_runway


def compute_debt_and_net_worth(profile: Dict[str, Any], annual_income: float) -> Dict[str, float]:
    assets = profile.get("assets", {})
    liabilities = profile.get("liabilities", {})

    total_debt_outstanding = _safe_float(liabilities.get("totalOutstanding"))
    total_asset = _safe_float(assets.get("totalAsset"))

    debt_to_income = total_debt_outstanding / annual_income if annual_income > 0 else 0.0

    net_worth = total_asset - total_debt_outstanding

    return {
        "total_debt_outstanding": total_debt_outstanding,
        "total_asset": total_asset,
        "debt_to_income": debt_to_income,
        "net_worth": net_worth,
    }


# ---------- Risk capacity & combination ----------

def compute_risk_capacity(profile: Dict[str, Any]) -> RiskCapacityResult:
    personal = profile.get("personal", {})

    age = _safe_float(personal.get("age"))
    profession_type = (personal.get("professionType") or "").strip()

    income_metrics = compute_income_metrics(profile)
    _, months_of_runway = compute_liquid_assets(profile)
    debt_metrics = compute_debt_and_net_worth(profile, income_metrics["annual_income"])

    # proxy: use debt outstanding / income if EMI fields are not correctly filled yet
    if income_metrics["annual_income"] > 0:
        emi_to_income = debt_metrics["total_debt_outstanding"] / income_metrics["annual_income"]
    else:
        emi_to_income = 0.0

    # Age score
    if age <= 0:
        age_score = 0
    elif age <= 30:
        age_score = 25
    elif age <= 40:
        age_score = 20
    elif age <= 50:
        age_score = 12
    elif age <= 60:
        age_score = 6
    else:
        age_score = 0

    # Job stability
    pt = profession_type.lower()
    if pt == "salaried":
        job_score = 20
    elif pt in {"selfemployed", "self-employed"}:
        job_score = 12
    elif pt in {"businessowner", "business owner"}:
        job_score = 10
    elif pt in {"student", "retired"}:
        job_score = 5
    else:
        job_score = 10

    # Emergency buffer
    if months_of_runway >= 12:
        buffer_score = 25
    elif months_of_runway >= 6:
        buffer_score = 18
    elif months_of_runway >= 3:
        buffer_score = 10
    else:
        buffer_score = 0

    # Debt burden: EMI / income
    if emi_to_income < 0.2:
        debt_score = 20
    elif emi_to_income <= 0.4:
        debt_score = 10
    else:
        debt_score = 0

    total = age_score + job_score + buffer_score + debt_score

    if total <= 30:
        band = "Low"
    elif total <= 60:
        band = "Medium"
    else:
        band = "High"

    return RiskCapacityResult(
        score=total,
        age_score=age_score,
        job_score=job_score,
        buffer_score=buffer_score,
        debt_score=debt_score,
        band=band,
    )


def combine_risk(profile: Dict[str, Any]) -> RiskResult:
    capacity = compute_risk_capacity(profile)
    risk_quiz = profile.get("riskQuiz") or {}
    quiz_score = _safe_float(risk_quiz.get("totalScore"))
    quiz_label_raw = (risk_quiz.get("riskLabel") or "").strip().lower()

    if not quiz_label_raw:
        # derive from score
        if quiz_score <= 10:
            quiz_label: RiskBand = "Conservative"
        elif quiz_score <= 20:
            quiz_label = "Balanced"
        else:
            quiz_label = "Aggressive"
    elif "conservative" in quiz_label_raw:
        quiz_label = "Conservative"
    elif "balanced" in quiz_label_raw:
        quiz_label = "Balanced"
    elif "aggressive" in quiz_label_raw:
        quiz_label = "Aggressive"
    else:
        quiz_label = "Balanced"

    # tolerance band
    if quiz_label == "Conservative":
        tol_band = "Low"
    elif quiz_label == "Balanced":
        tol_band = "Medium"
    else:
        tol_band = "High"

    # Final band = safer of capacity and tolerance
    if capacity.band == "Low" or tol_band == "Low":
        final_band: RiskBand = "Conservative"
    elif capacity.band == "High" and tol_band == "High":
        final_band = "Aggressive"
    else:
        final_band = "Balanced"

    return RiskResult(
        capacity=capacity,
        quiz_score=quiz_score,
        quiz_label=quiz_label,
        final_band=final_band,
    )


# ---------- Current allocation ----------

def compute_current_allocation(profile: Dict[str, Any]) -> Allocation:
    assets = profile.get("assets", {})

    equity = sum(
        _safe_float(assets.get(k))
        for k in ["directEquity", "equityMf", "pms", "aif", "esops", "reitEquityPortion", "startupInvestments"]
    )
    debt = sum(
        _safe_float(assets.get(k))
        for k in ["debtMf", "bonds", "fds", "rds", "postOfficeSchemes", "ppfDebtPortion", "npsDebtPortion"]
    )
    gold = _safe_float(assets.get("goldSilver"))
    cash = sum(_safe_float(assets.get(k)) for k in ["bankBalance1", "bankBalance2", "shortTermDeposits"])
    real_estate_alt = sum(
        _safe_float(assets.get(k))
        for k in [
            "selfOccupiedHouse",
            "house2",
            "realEstate",
            "reitOtherPortion",
            "crypto",
            "startupInvestmentsAltPortion",
        ]
    )

    total = equity + debt + gold + cash + real_estate_alt
    if total <= 0:
        total = 1.0  # avoid division by zero

    return Allocation(
        equity=equity,
        debt=debt,
        gold=gold,
        cash=cash,
        real_estate_alt=real_estate_alt,
        total=total,
    )


# ---------- Plan templates ----------

def _base_template(plan_id: str) -> Dict[str, Tuple[float, float]]:
    if plan_id == "safety":
        return {
            "equity": (0.20, 0.30),
            "debt": (0.50, 0.60),
            "gold": (0.05, 0.10),
            "cash": (0.05, 0.10),
        }
    if plan_id == "balanced":
        return {
            "equity": (0.45, 0.60),
            "debt": (0.25, 0.40),
            "gold": (0.05, 0.10),
            "cash": (0.03, 0.05),
        }
    # growth
    return {
        "equity": (0.70, 0.85),
        "debt": (0.10, 0.20),
        "gold": (0.05, 0.10),
        "cash": (0.00, 0.05),
    }


def _pick_percent(min_p: float, max_p: float, final_band: RiskBand) -> float:
    if final_band == "Conservative":
        return min_p
    if final_band == "Aggressive":
        return max_p
    return (min_p + max_p) / 2.0


def build_plan(
    plan_id: str,
    allocation: Allocation,
    risk: RiskResult,
    income_metrics: Dict[str, float],
) -> PlanOutput:
    base = _base_template(plan_id)
    total = allocation.total

    target_pct: Dict[str, float] = {}
    target_val: Dict[str, float] = {}
    deltas: Dict[str, float] = {}

    for asset_class in ["equity", "debt", "gold", "cash"]:
        min_p, max_p = base[asset_class]
        p = _pick_percent(min_p, max_p, risk.final_band)
        target_pct[asset_class] = round(p * 100, 1)
        v = p * total
        target_val[asset_class] = round(v, 0)

        current = getattr(allocation, asset_class)
        deltas[asset_class] = round(v - current, 0)

    current_pct = {
        "equity": round(allocation.equity / total * 100, 1),
        "debt": round(allocation.debt / total * 100, 1),
        "gold": round(allocation.gold / total * 100, 1),
        "cash": round(allocation.cash / total * 100, 1),
        "real_estate_alt": round(allocation.real_estate_alt / total * 100, 1),
    }

    # Names & prices
    if plan_id == "safety":
        name = "Safety plan (low risk)"
        price = 1000
    elif plan_id == "balanced":
        name = "Balanced plan"
        price = 3000
    else:
        name = "Growth plan (high risk)"
        price = 5000

    monthly_savings = income_metrics.get("monthly_savings", 0.0)

    if plan_id == "safety":
        summary = "Lower volatility, debt-heavy portfolio designed to protect capital."
    elif plan_id == "balanced":
        summary = "Blend of growth and stability with a mix of equity and debt."
    else:
        summary = "Equity-heavy portfolio focused on long-term growth and higher risk."

    bullets: List[str] = []

    delta_eq = deltas["equity"]
    delta_debt = deltas["debt"]
    if delta_eq < 0 and delta_debt > 0:
        bullets.append(
            f"Reduce equity by about ₹{abs(int(delta_eq)):,} and increase debt by ₹{int(delta_debt):,}."
        )
    elif delta_eq > 0 and delta_debt < 0:
        bullets.append(
            f"Increase equity by about ₹{int(delta_eq):,} funded by reducing debt by ₹{abs(int(delta_debt)):,}."
        )

    delta_gold = deltas["gold"]
    if delta_gold < 0:
        bullets.append(
            f"Trim gold holdings by approximately ₹{abs(int(delta_gold)):,} to keep gold around {target_pct['gold']}% of the portfolio."
        )
    elif delta_gold > 0:
        bullets.append(
            f"Add around ₹{int(delta_gold):,} to gold (or gold ETFs) to reach ~{target_pct['gold']}% allocation."
        )

    if monthly_savings > 0:
        bullets.append(
            f"Invest roughly ₹{int(monthly_savings):,} per month towards this plan until the allocation is reached."
        )
    else:
        bullets.append(
            "Current cash flow shows little or no monthly surplus; prioritise building savings and reducing EMIs before aggressive investing."
        )

    bullets.append(f"Plan built for a {risk.final_band.lower()}-risk profile ({risk.quiz_label} attitude).")

    return PlanOutput(
        id=plan_id,
        name=name,
        recommended=False,
        target_allocation_pct=target_pct,
        target_allocation_value=target_val,
        current_allocation_pct=current_pct,
        deltas_value=deltas,
        summary=summary,
        bullets=bullets,
        price=price,
    )


def generate_plans(profile: Dict[str, Any]) -> Dict[str, Any]:
    income_metrics = compute_income_metrics(profile)
    allocation = compute_current_allocation(profile)
    risk = combine_risk(profile)

    plans: List[PlanOutput] = []
    for pid in ["safety", "balanced", "growth"]:
        plans.append(build_plan(pid, allocation, risk, income_metrics))

    recommended_id = {
        "Conservative": "safety",
        "Balanced": "balanced",
        "Aggressive": "growth",
    }[risk.final_band]

    for p in plans:
        if p.id == recommended_id:
            p.recommended = True

    def plan_to_dict(p: PlanOutput) -> Dict[str, Any]:
        return {
            "id": p.id,
            "name": p.name,
            "recommended": p.recommended,
            "price": p.price,
            "summary": p.summary,
            "targetAllocationPct": p.target_allocation_pct,
            "targetAllocationValue": p.target_allocation_value,
            "currentAllocationPct": p.current_allocation_pct,
            "deltasValue": p.deltas_value,
            "bullets": p.bullets,
        }

    result = {
        "profileSummary": {
            "age": profile.get("personal", {}).get("age"),
            "city": profile.get("personal", {}).get("city"),
            "annualIncome": income_metrics["annual_income"],
            "monthlySavings": income_metrics["monthly_savings"],
        },
        "risk": {
            "capacityScore": risk.capacity.score,
            "capacityBand": risk.capacity.band,
            "quizScore": risk.quiz_score,
            "quizLabel": risk.quiz_label,
            "finalBand": risk.final_band,
        },
        "currentAllocation": {
            "equityPct": round(allocation.equity / allocation.total * 100, 1),
            "debtPct": round(allocation.debt / allocation.total * 100, 1),
            "goldPct": round(allocation.gold / allocation.total * 100, 1),
            "cashPct": round(allocation.cash / allocation.total * 100, 1),
            "realEstateAltPct": round(allocation.real_estate_alt / allocation.total * 100, 1),
        },
        "plans": [plan_to_dict(p) for p in plans],
    }

    return result

def generate_financial_blueprint(profile: Dict[str, Any]) -> Dict[str, Any]:
    return build_financial_blueprint(profile)
