# backend/app/planner/emi.py
from typing import Any, Dict, Iterable, Optional


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


def _resolve_emi(
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


def get_total_monthly_emi(profile: Dict[str, Any]) -> float:
    liabilities = profile.get("liabilities", {}) or {}
    cash_flow = profile.get("cashFlow", {}) or {}

    emi_values = [
        _resolve_emi(
            liabilities,
            cash_flow,
            "houseLoan1Emi",
            legacy_liability_keys=["homeLoan1Emi"],
            legacy_cashflow_keys=["homeEmi1"],
        ),
        _resolve_emi(
            liabilities,
            cash_flow,
            "houseLoan2Emi",
            legacy_liability_keys=["homeLoan2Emi"],
            legacy_cashflow_keys=["homeEmi2"],
        ),
        _resolve_emi(
            liabilities,
            cash_flow,
            "personalLoan1Emi",
            legacy_liability_keys=["personalLoanEmi"],
            legacy_cashflow_keys=["personalLoanEmi"],
        ),
        _resolve_emi(
            liabilities,
            cash_flow,
            "personalLoan2Emi",
        ),
        _resolve_emi(
            liabilities,
            cash_flow,
            "vehicleLoanEmi",
            legacy_cashflow_keys=["vehicleLoanEmi"],
        ),
        _resolve_emi(
            liabilities,
            cash_flow,
            "loanAgainstSharesEmi",
        ),
        _resolve_emi(
            liabilities,
            cash_flow,
            "creditCard1Emi",
            legacy_liability_keys=["creditCardEmi"],
            legacy_cashflow_keys=["creditCardEmi"],
        ),
        _resolve_emi(
            liabilities,
            cash_flow,
            "creditCard2Emi",
        ),
        _resolve_emi(
            liabilities,
            cash_flow,
            "othersEmi",
            legacy_liability_keys=["otherLoanEmi"],
            legacy_cashflow_keys=["otherEmi"],
        ),
    ]

    return sum(emi_values)
