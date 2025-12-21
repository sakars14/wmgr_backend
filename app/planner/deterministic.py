# backend/app/planner/deterministic.py
from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional

from .emi import get_total_monthly_emi

BLUEPRINT_VERSION = "phase1-blueprint-v1"
PROFILE_SCHEMA_VERSION = "phase0-v2"

ASSET_COMPONENT_KEYS = [
    "selfOccupiedHouse",
    "house2",
    "bankBalance1",
    "bankBalance2",
    "directEquity",
    "equityMf",
    "debtMf",
    "bonds",
    "goldSilver",
    "esops",
    "reit",
    "realEstate",
    "pms",
    "aif",
    "crypto",
    "startupInvestments",
    "fds",
    "rds",
    "postOfficeSchemes",
    "ppf",
    "nps",
    "traditionalInsurance",
    "vehicle",
    "rentalDeposit",
    "others",
]

LIABILITY_OUTSTANDING_KEYS = [
    "houseLoan1",
    "houseLoan2",
    "loanAgainstShares",
    "personalLoan1",
    "personalLoan2",
    "creditCard1",
    "creditCard2",
    "vehicleLoan",
    "others",
    "totalOutstanding",
]

LIABILITY_OUTSTANDING_COMPONENT_KEYS = [
    key for key in LIABILITY_OUTSTANDING_KEYS if key != "totalOutstanding"
]

LIABILITY_REMAINING_KEYS = [
    "houseLoan1RemainingMonths",
    "houseLoan2RemainingMonths",
    "loanAgainstSharesRemainingMonths",
    "personalLoan1RemainingMonths",
    "personalLoan2RemainingMonths",
    "creditCard1RemainingMonths",
    "creditCard2RemainingMonths",
    "vehicleLoanRemainingMonths",
    "othersRemainingMonths",
]

CONTRIBUTION_FIELDS = [
    "sipEquityMfMonthly",
    "sipDebtMfMonthly",
    "sipDirectEquityMonthly",
    "ppfMonthly",
    "npsMonthly",
    "epfMonthly",
    "otherInvestMonthly",
]

INSURANCE_FIELDS = [
    "termPolicyClient",
    "officeHealthInsuranceClient",
    "personalHealthBaseInsuranceClient",
    "superTopUpHealthInsuranceClient",
    "termPolicySpouse",
    "officeHealthInsuranceSpouse",
    "personalHealthBaseInsuranceSpouse",
    "superTopUpHealthInsuranceSpouse",
    "otherPolicy",
]

INSURANCE_PREMIUM_FIELDS = [
    f"{key}PremiumPerYear" for key in INSURANCE_FIELDS
]

GOAL_FIELDS = [
    {"key": "child1UnderGraduateEducation", "label": "Child 1 Under Graduate Education"},
    {"key": "child2UnderGraduateEducation", "label": "Child 2 Under Graduate Education"},
    {"key": "child1PostGraduateEducation", "label": "Child 1 Post Graduate Education"},
    {"key": "child2PostGraduateEducation", "label": "Child 2 Post Graduate Education"},
    {"key": "child1Marriage", "label": "Child 1 Marriage"},
    {"key": "child2Marriage", "label": "Child 2 Marriage"},
    {"key": "retirement", "label": "Retirement"},
    {"key": "house", "label": "House"},
    {"key": "startBusiness", "label": "Start Business"},
    {"key": "car", "label": "Car"},
    {"key": "gold", "label": "Gold"},
    {"key": "vacation", "label": "Vacation"},
    {"key": "others", "label": "Others"},
    {"key": "totalGoalValue", "label": "Total Goal Value"},
]

GOAL_ENTRY_FIELDS = [
    {
        "key": entry["key"],
        "label": entry["label"],
        "horizonKey": f"{entry['key']}HorizonYears",
        "priorityKey": f"{entry['key']}Priority",
    }
    for entry in GOAL_FIELDS
    if entry["key"] != "totalGoalValue"
]


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


def _is_blank(v: Any) -> bool:
    return v is None or (isinstance(v, str) and v.strip() == "")


def _now_iso() -> str:
    return datetime.utcnow().isoformat() + "Z"


def _warn_if_blank(warnings: List[str], path: str, raw_value: Any, assumed: str = "0") -> None:
    if _is_blank(raw_value):
        warnings.append(f"Missing {path} (assumed {assumed}).")


def _warn_mismatch(
    warnings: List[str],
    label: str,
    provided_total: float,
    computed_total: float,
) -> None:
    if provided_total <= 0:
        return
    delta = abs(computed_total - provided_total)
    if delta / provided_total > 0.05:
        warnings.append(
            f"{label} mismatch: computed={computed_total:.2f}, provided={provided_total:.2f}."
        )


def _resolve_emi_value(
    liabilities: Dict[str, Any],
    cash_flow: Dict[str, Any],
    canonical_key: str,
    legacy_liability_keys: Optional[Iterable[str]] = None,
    legacy_cashflow_keys: Optional[Iterable[str]] = None,
) -> float:
    val = _safe_float(liabilities.get(canonical_key))
    if val > 0:
        return val
    for key in legacy_liability_keys or []:
        val = _safe_float(liabilities.get(key))
        if val > 0:
            return val
    for key in legacy_cashflow_keys or []:
        val = _safe_float(cash_flow.get(key))
        if val > 0:
            return val
    return 0.0


def compute_surplus(profile: Dict[str, Any]) -> Dict[str, float]:
    """
    Uses Phase-0 v2 keys (with fallback to legacy cashFlow EMI fields):
      - personal.monthlyIncomeInHand
      - personal.annualBonus
      - personal.passiveIncome
      - personal.otherIncome1 / otherIncome2
      - cashFlow.totalMonthlyExpense
      - liabilities.<emi fields>
    """
    personal = profile.get("personal", {}) or {}
    cash_flow = profile.get("cashFlow", {}) or {}

    monthly_income_in_hand = _safe_float(personal.get("monthlyIncomeInHand"))
    annual_bonus = _safe_float(personal.get("annualBonus"))
    passive_income = _safe_float(personal.get("passiveIncome"))
    other1 = _safe_float(personal.get("otherIncome1"))
    other2 = _safe_float(personal.get("otherIncome2"))

    annual_income = (
        12 * monthly_income_in_hand
        + annual_bonus
        + 12 * passive_income
        + 12 * other1
        + 12 * other2
    )
    monthly_income_total = annual_income / 12.0

    total_monthly_expense = _safe_float(cash_flow.get("totalMonthlyExpense"))

    total_monthly_emi = get_total_monthly_emi(profile)

    monthly_surplus = monthly_income_total - total_monthly_expense - total_monthly_emi

    return {
        "annualIncome": annual_income,
        "monthlyIncomeTotal": monthly_income_total,
        "totalMonthlyExpense": total_monthly_expense,
        "totalMonthlyEmi": total_monthly_emi,
        "monthlySurplus": monthly_surplus,
    }


def build_financial_blueprint(profile: Dict[str, Any]) -> Dict[str, Any]:
    warnings: List[str] = []

    personal = profile.get("personal", {}) or {}
    cash_flow = profile.get("cashFlow", {}) or {}
    assets = profile.get("assets", {}) or {}
    liabilities = profile.get("liabilities", {}) or {}
    contributions = profile.get("contributions", {}) or {}
    emergency = profile.get("emergency", {}) or {}
    insurance = profile.get("insurance", {}) or {}
    goals = profile.get("goals", {}) or {}
    risk_quiz = profile.get("riskQuiz", {}) or {}

    if profile.get("schemaVersion") != PROFILE_SCHEMA_VERSION:
        warnings.append(
            "profile.schemaVersion is not phase0-v2; blueprint assumes phase0-v2 inputs."
        )

    _warn_if_blank(warnings, "personal.monthlyIncomeInHand", personal.get("monthlyIncomeInHand"))
    _warn_if_blank(warnings, "cashFlow.totalMonthlyExpense", cash_flow.get("totalMonthlyExpense"))
    _warn_if_blank(warnings, "emergency.monthsTarget", emergency.get("monthsTarget"))

    surplus = compute_surplus(profile)

    annual_large_expenses = _safe_float(cash_flow.get("annualLargeExpenses"))

    monthly_total_outflow = surplus["totalMonthlyExpense"] + surplus["totalMonthlyEmi"]

    months_target = _safe_float(emergency.get("monthsTarget"))
    dedicated_amount = _safe_float(emergency.get("dedicatedAmount"))
    target_corpus = monthly_total_outflow * months_target
    current_dedicated = dedicated_amount
    emergency_gap = max(0.0, target_corpus - current_dedicated)
    emergency_surplus = max(0.0, current_dedicated - target_corpus)
    emergency_status = "Underfunded" if emergency_gap > 0 else "Adequate"

    contributions_breakdown = {
        key: _safe_float(contributions.get(key)) for key in CONTRIBUTION_FIELDS
    }
    contributions_sum = sum(contributions_breakdown.values())

    insurance_covers = {
        key: _safe_float(insurance.get(key)) for key in INSURANCE_FIELDS
    }
    insurance_premiums = {
        key: _safe_float(insurance.get(key)) for key in INSURANCE_PREMIUM_FIELDS
    }

    goal_amounts = {entry["key"]: _safe_float(goals.get(entry["key"])) for entry in GOAL_FIELDS}
    goal_horizons = {
        entry["key"]: _safe_float(goals.get(entry["horizonKey"]))
        for entry in GOAL_ENTRY_FIELDS
    }
    goal_priorities = {
        entry["key"]: None
        if _is_blank(goals.get(entry["priorityKey"]))
        else str(goals.get(entry["priorityKey"]))
        for entry in GOAL_ENTRY_FIELDS
    }

    goal_items: List[Dict[str, Any]] = []
    for entry in GOAL_ENTRY_FIELDS:
        amount_today = _safe_float(goals.get(entry["key"]))
        horizon_years = _safe_float(goals.get(entry["horizonKey"]))
        priority = (
            None if _is_blank(goals.get(entry["priorityKey"])) else str(goals.get(entry["priorityKey"]))
        )
        if amount_today > 0 or horizon_years > 0 or priority:
            goal_items.append(
                {
                    "key": entry["key"],
                    "label": entry["label"],
                    "amountToday": amount_today,
                    "horizonYears": horizon_years,
                    "priority": priority,
                }
            )

    user_total_asset_raw = assets.get("totalAsset")
    user_total_asset = _safe_float(user_total_asset_raw)
    computed_assets_total = sum(
        _safe_float(assets.get(key)) for key in ASSET_COMPONENT_KEYS
    )
    _warn_mismatch(
        warnings,
        "Asset total",
        user_total_asset,
        computed_assets_total,
    )
    assets_total = (
        computed_assets_total if computed_assets_total > 0 else user_total_asset
    )

    user_total_outstanding_raw = liabilities.get("totalOutstanding")
    user_total_outstanding = _safe_float(user_total_outstanding_raw)
    computed_liabilities_total = sum(
        _safe_float(liabilities.get(key)) for key in LIABILITY_OUTSTANDING_COMPONENT_KEYS
    )
    _warn_mismatch(
        warnings,
        "Liability total",
        user_total_outstanding,
        computed_liabilities_total,
    )
    liabilities_total = (
        computed_liabilities_total if computed_liabilities_total > 0 else user_total_outstanding
    )

    emi_breakdown = {
        "houseLoan1Emi": _resolve_emi_value(
            liabilities,
            cash_flow,
            "houseLoan1Emi",
            legacy_liability_keys=["homeLoan1Emi"],
            legacy_cashflow_keys=["homeEmi1"],
        ),
        "houseLoan2Emi": _resolve_emi_value(
            liabilities,
            cash_flow,
            "houseLoan2Emi",
            legacy_liability_keys=["homeLoan2Emi"],
            legacy_cashflow_keys=["homeEmi2"],
        ),
        "personalLoan1Emi": _resolve_emi_value(
            liabilities,
            cash_flow,
            "personalLoan1Emi",
            legacy_liability_keys=["personalLoanEmi"],
            legacy_cashflow_keys=["personalLoanEmi"],
        ),
        "personalLoan2Emi": _resolve_emi_value(
            liabilities,
            cash_flow,
            "personalLoan2Emi",
        ),
        "vehicleLoanEmi": _resolve_emi_value(
            liabilities,
            cash_flow,
            "vehicleLoanEmi",
            legacy_cashflow_keys=["vehicleLoanEmi"],
        ),
        "loanAgainstSharesEmi": _resolve_emi_value(
            liabilities,
            cash_flow,
            "loanAgainstSharesEmi",
        ),
        "creditCard1Emi": _resolve_emi_value(
            liabilities,
            cash_flow,
            "creditCard1Emi",
            legacy_liability_keys=["creditCardEmi"],
            legacy_cashflow_keys=["creditCardEmi"],
        ),
        "creditCard2Emi": _resolve_emi_value(
            liabilities,
            cash_flow,
            "creditCard2Emi",
        ),
        "othersEmi": _resolve_emi_value(
            liabilities,
            cash_flow,
            "othersEmi",
            legacy_liability_keys=["otherLoanEmi"],
            legacy_cashflow_keys=["otherEmi"],
        ),
    }

    remaining_months = {
        key: _safe_float(liabilities.get(key)) for key in LIABILITY_REMAINING_KEYS
    }

    outstanding = {
        key: _safe_float(liabilities.get(key)) for key in LIABILITY_OUTSTANDING_KEYS
    }

    risk_quiz_label = risk_quiz.get("riskLabel")
    risk_quiz_score = _safe_float(risk_quiz.get("totalScore"))
    risk_tolerance_self = personal.get("riskToleranceSelf")
    mismatch = False
    if not _is_blank(risk_quiz_label) and not _is_blank(risk_tolerance_self):
        mismatch = str(risk_quiz_label).strip().lower() != str(risk_tolerance_self).strip().lower()

    inputs_assets: Dict[str, Any] = {
        "computedAssetsTotal": computed_assets_total,
    }
    if not _is_blank(user_total_asset_raw):
        inputs_assets["userProvidedTotalAsset"] = user_total_asset

    out = {
        "meta": {
            "blueprintVersion": BLUEPRINT_VERSION,
            "generatedAt": _now_iso(),
            "profileSchemaVersion": PROFILE_SCHEMA_VERSION,
            "source": "deterministic",
        },
        "inputsSnapshot": {
            "personal": {
                "monthlyIncomeInHand": _safe_float(personal.get("monthlyIncomeInHand")),
                "annualBonus": _safe_float(personal.get("annualBonus")),
                "passiveIncome": _safe_float(personal.get("passiveIncome")),
                "otherIncome1": _safe_float(personal.get("otherIncome1")),
                "otherIncome2": _safe_float(personal.get("otherIncome2")),
                "riskToleranceSelf": risk_tolerance_self,
            },
            "cashFlow": {
                "totalMonthlyExpense": surplus["totalMonthlyExpense"],
                "annualLargeExpenses": annual_large_expenses,
            },
            "liabilities": {
                "outstanding": outstanding,
                "emi": emi_breakdown,
                "remainingMonths": remaining_months,
            },
            "assets": inputs_assets,
            "contributions": {
                "sumMonthly": contributions_sum,
                "breakdown": contributions_breakdown,
            },
            "emergency": {
                "monthsTarget": months_target,
                "dedicatedAmount": dedicated_amount,
            },
            "insurance": {
                "covers": insurance_covers,
                "premiums": insurance_premiums,
            },
            "goals": {
                "amounts": goal_amounts,
                "horizonYears": goal_horizons,
                "priority": goal_priorities,
            },
            "risk": {
                "quizLabel": risk_quiz_label,
                "quizScore": risk_quiz_score,
                "riskToleranceSelf": risk_tolerance_self,
            },
        },
        "derived": {
            "income": {
                "monthlyTotal": surplus["monthlyIncomeTotal"],
                "annualTotal": surplus["annualIncome"],
            },
            "outflow": {
                "monthlyExpenseExclEmi": surplus["totalMonthlyExpense"],
                "monthlyEmiTotal": surplus["totalMonthlyEmi"],
                "monthlyTotalOutflow": monthly_total_outflow,
            },
            "surplus": {
                "monthlySurplus": surplus["monthlySurplus"],
            },
            "netWorth": {
                "assetsTotal": assets_total,
                "liabilitiesTotal": liabilities_total,
                "netWorth": assets_total - liabilities_total,
            },
            "emergency": {
                "targetMonths": months_target,
                "targetCorpus": target_corpus,
                "currentDedicated": current_dedicated,
                "gap": emergency_gap,
                "surplus": emergency_surplus,
                "status": emergency_status,
            },
            "goals": goal_items,
            "risk": {
                "riskQuizLabel": risk_quiz_label,
                "riskToleranceSelf": risk_tolerance_self,
                "mismatch": mismatch,
            },
        },
        "warnings": warnings,
    }
    return out
