from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Literal, Optional, Tuple

from pydantic import BaseModel, ValidationError

try:
    from openai import OpenAI
except Exception:  # pragma: no cover - runtime dependency
    OpenAI = None

NARRATION_VERSION = "phase2-narrator-v1"
DEFAULT_MODEL = "gpt-5.2-mini"
DEFAULT_TEMPERATURE = 0.2

_NUMBER_RE = re.compile(r"\d+(?:\.\d+)?")
_NON_NUMERIC_RE = re.compile(r"[^0-9.\-]+")


class PlanSummary(BaseModel):
    oneLiner: str
    healthScoreLabel: Optional[str] = None
    riskBand: str

    class Config:
        extra = "forbid"


class NarrationSection(BaseModel):
    title: str
    markdown: str

    class Config:
        extra = "forbid"


class ActionChecklistItem(BaseModel):
    id: str
    title: str
    why: str
    priority: Literal["High", "Medium", "Low"]

    class Config:
        extra = "forbid"


class ClarifyingQuestion(BaseModel):
    id: str
    question: str
    whyItMatters: str

    class Config:
        extra = "forbid"


class NarrationResponse(BaseModel):
    narrationVersion: Literal["phase2-narrator-v1"] = NARRATION_VERSION
    generatedAt: str
    planSummary: PlanSummary
    sections: List[NarrationSection]
    actionChecklist: List[ActionChecklistItem]
    clarifyingQuestions: List[ClarifyingQuestion]
    disclosures: List[str]

    class Config:
        extra = "forbid"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _system_prompt() -> str:
    return (
        "You are a financial plan narrator that explains an existing blueprint and plan. "
        "Rules:\n"
        "- Do NOT perform calculations.\n"
        "- Do NOT introduce any numeric values not present in the blueprint or plan JSON.\n"
        "- If a number is required but missing, ask a clarifying question instead.\n"
        "- If blueprint.warnings is non-empty, include clarifyingQuestions that resolve those warnings.\n"
        "- Output must be valid JSON matching the NarrationResponse schema ONLY.\n"
        "- Do not include markdown fences, commentary, or extra keys.\n"
        "\n"
        "Grounded numbers requirement:\n"
        "- Include at least 6 numeric facts copied verbatim from the provided JSON.\n"
        "- The numeric facts must include:\n"
        "  a) blueprint.derived.outflow.monthlyEmiTotal\n"
        "  b) blueprint.derived.outflow.monthlyExpenseExclEmi\n"
        "  c) blueprint.derived.surplus.monthlySurplus\n"
        "  d) blueprint.derived.emergency.targetMonths\n"
        "  e) blueprint.derived.emergency.targetCorpus\n"
        "  f) blueprint.derived.emergency.gap\n"
        "- If any are missing/null/0 due to missing inputs, add clarifyingQuestions with specific fields and why.\n"
        "\n"
        "Specific references requirement:\n"
        "- Reference at least one concrete liability (key or label) with EMI and remaining months if present.\n"
        "- Reference at least two concrete goals using blueprint.derived.goalProjections.\n"
        "- For the top 2 goals, include: futureCostInflated, assumedReturnAnnual, requiredMonthlyInvestmentRounded, status, and shortfallMonthlyRounded (if any).\n"
        "- assumedReturnAnnual is provided as a decimal in goalProjections; copy it verbatim (do not convert).\n"
        "- Also state assumptions: inflation rate and the return basis (debt/blended/equity) from blueprint.meta.assumptions.\n"
        "- If fewer exist, reference all available and ask for the missing ones.\n"
        "\n"
        "Section requirements (minimum 4 sections with exact titles):\n"
        "- Cash flow snapshot\n"
        "- Emergency buffer\n"
        "- Goals & timeline\n"
        "- Debt & next 30 days\n"
        "\n"
        "Emergency buffer wording:\n"
        "- Use targetMonths, targetCorpus, gap, surplus, status.\n"
        "- If status is Adequate and surplus > 0, say \"You are above target by \u20B9X\" using the exact surplus value.\n"
        "- If status is Underfunded, say \"You are short by \u20B9X\" using the exact gap value.\n"
        "- If status is Unknown, ask clarifying questions and avoid stating adequacy.\n"
        "- Use INR formatting (₹ with Indian commas) for currency amounts.\n"
        "- Do not output debug strings like \"gap=None\" or \"status=None\".\n"
        "\n"
        "Action checklist:\n"
        "- Must contain 3-5 items with priorities.\n"
        "- Convert checklistSeeds into actionChecklist items without changing numbers.\n"
        "\n"
        "No generic filler:\n"
        "- Do not output generic phrases without client-specific details.\n"
        "- Every section must contain at least one concrete number or named item (loan/goal) OR a clarifying question explaining missing data.\n"
        "\n"
        "NarrationResponse schema:\n"
        "{\n"
        '  "narrationVersion": "phase2-narrator-v1",\n'
        '  "generatedAt": "<iso>",\n'
        '  "planSummary": {"oneLiner": "string", "healthScoreLabel": "string (optional)", "riskBand": "string"},\n'
        '  "sections": [{"title": "string", "markdown": "string"}],\n'
        '  "actionChecklist": [{"id": "string", "title": "string", "why": "string", "priority": "High|Medium|Low"}],\n'
        '  "clarifyingQuestions": [{"id": "string", "question": "string", "whyItMatters": "string"}],\n'
        '  "disclosures": ["string"]\n'
        "}\n"
    )


def _user_prompt(
    blueprint: Dict[str, Any],
    plan: Dict[str, Any],
    facts: Dict[str, Any],
    checklist_seeds: List[Dict[str, Any]],
) -> str:
    facts_json = json.dumps(facts, ensure_ascii=True, separators=(",", ":"))
    seeds_json = json.dumps(checklist_seeds, ensure_ascii=True, separators=(",", ":"))
    blueprint_json = json.dumps(blueprint, ensure_ascii=True, separators=(",", ":"))
    plan_json = json.dumps(plan, ensure_ascii=True, separators=(",", ":"))
    return (
        "Context: locale=India, currency=INR.\n"
        "Use facts first, then blueprint/plan for detail. Facts.topGoals are sourced from derived.goalProjections.\n"
        "ChecklistSeeds JSON:\n"
        f"{seeds_json}\n"
        "Facts JSON:\n"
        f"{facts_json}\n"
        "Blueprint JSON:\n"
        f"{blueprint_json}\n"
        "Plan JSON:\n"
        f"{plan_json}\n"
        "Return NarrationResponse JSON only."
    )


def _extract_response_text(resp: Any) -> str:
    if resp is None:
        return ""
    text = getattr(resp, "output_text", None)
    if text:
        return text
    choices = getattr(resp, "choices", None)
    if choices:
        msg = getattr(choices[0], "message", None)
        if msg and getattr(msg, "content", None):
            return msg.content
    output = getattr(resp, "output", None)
    if isinstance(output, list):
        for item in output:
            content = getattr(item, "content", None)
            if isinstance(content, list):
                for part in content:
                    if getattr(part, "type", None) == "output_text":
                        return getattr(part, "text", "")
    return ""


def _call_openai(system: str, user: str, model: str, temperature: float) -> str:
    if OpenAI is None:
        return ""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return ""
    client = OpenAI(api_key=api_key)

    try:
        resp = client.responses.create(
            model=model,
            input=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=temperature,
        )
        return _extract_response_text(resp)
    except Exception:
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                temperature=temperature,
            )
            return _extract_response_text(resp)
        except Exception:
            return ""


def _parse_json(raw: str) -> Optional[Dict[str, Any]]:
    if not raw:
        return None
    text = raw.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            try:
                return json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                return None
    return None


def _alpha_suffix(idx: int) -> str:
    if idx < 0:
        idx = 0
    letters: List[str] = []
    while True:
        idx, rem = divmod(idx, 26)
        letters.append(chr(ord("a") + rem))
        if idx == 0:
            break
        idx -= 1
    return "".join(reversed(letters))


def _warning_to_question(warning: str) -> Tuple[str, str]:
    w = (warning or "").strip()
    w_lower = w.lower()
    if w_lower.startswith("missing "):
        path = w[8:].split("(")[0].strip().rstrip(".")
        question = f"Please confirm the value for {path}."
        why = "This input is required to avoid assumptions in your report."
        return question, why
    if "mismatch" in w_lower:
        label = w.split("mismatch", 1)[0].strip().rstrip(":")
        question = f"Please confirm the correct {label.lower()} values."
        why = "The report must reflect your actual totals without assumptions."
        return question, why
    if "schema" in w_lower:
        question = "Can you confirm the profile schema version used for this plan?"
        why = "The report relies on consistent field definitions."
        return question, why
    question = "Can you confirm the inputs behind this warning?"
    why = "This helps ensure the report reflects your actual data."
    return question, why


def _warnings_to_questions(warnings: Iterable[str]) -> List[ClarifyingQuestion]:
    questions: List[ClarifyingQuestion] = []
    for idx, warning in enumerate(warnings):
        question, why = _warning_to_question(warning)
        questions.append(
            ClarifyingQuestion(
                id=f"clarify-{_alpha_suffix(idx)}",
                question=question,
                whyItMatters=why,
            )
        )
    return questions


def _safe_float(val: Any) -> Optional[float]:
    if val is None:
        return None
    try:
        if isinstance(val, bool):
            return None
        if isinstance(val, (int, float)):
            return float(val)
        if isinstance(val, str) and val.strip() == "":
            return None
        return float(val)
    except (TypeError, ValueError):
        return None


def _safe_number_text(value: Any) -> str:
    num = _safe_float(value)
    if num is None:
        num = 0.0
    if float(num).is_integer():
        return str(int(num))
    return str(num)


def _format_indian_int(value: int) -> str:
    s = str(abs(value))
    if len(s) <= 3:
        grouped = s
    else:
        last3 = s[-3:]
        rest = s[:-3]
        parts: List[str] = []
        while len(rest) > 2:
            parts.insert(0, rest[-2:])
            rest = rest[:-2]
        if rest:
            parts.insert(0, rest)
        grouped = ",".join(parts + [last3])
    if value < 0:
        grouped = "-" + grouped
    return grouped


def format_inr(value: Any) -> str:
    num = _safe_float(value)
    if num is None:
        num = 0.0
    negative = num < 0
    num_abs = abs(num)
    if float(num_abs).is_integer():
        int_part = int(round(num_abs))
        frac_part = ""
    else:
        raw = f"{num_abs:.2f}".rstrip("0").rstrip(".")
        if "." in raw:
            int_str, frac_part = raw.split(".", 1)
        else:
            int_str, frac_part = raw, ""
        int_part = int(int_str)
    grouped = _format_indian_int(int_part)
    if frac_part:
        grouped = f"{grouped}.{frac_part}"
    if negative:
        return f"-\u20B9{grouped}"
    return f"\u20B9{grouped}"


def _get_path(data: Dict[str, Any], path: List[str], default: Any = None) -> Any:
    cur: Any = data
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def _humanize_key(key: str) -> str:
    if not key:
        return ""
    label = re.sub(r"([a-z])([A-Z])", r"\1 \2", key)
    label = re.sub(r"(\D)(\d+)", r"\1 \2", label)
    return label.replace("_", " ").strip().title()


def _priority_rank(priority: Any) -> Tuple[int, str]:
    if priority is None:
        return (3, "")
    p = str(priority).strip().lower()
    if p.startswith("high"):
        return (0, "High")
    if p.startswith("medium"):
        return (1, "Medium")
    if p.startswith("low"):
        return (2, "Low")
    return (3, str(priority))


def _build_facts(blueprint: Dict[str, Any]) -> Dict[str, Any]:
    derived = blueprint.get("derived") or {}
    inputs = blueprint.get("inputsSnapshot") or {}

    goals = derived.get("goals") or []
    goal_projections = derived.get("goalProjections") or []
    goal_items: List[Dict[str, Any]] = []

    source_goals = goal_projections if goal_projections else goals
    for g in source_goals:
        if not isinstance(g, dict):
            continue
        rank, normalized = _priority_rank(g.get("priority"))
        horizon_value = _safe_float(g.get("horizonYears"))
        goal_items.append(
            {
                "key": g.get("goalKey") or g.get("key"),
                "label": g.get("label"),
                "amountToday": g.get("amountToday"),
                "horizonYears": g.get("horizonYears"),
                "priority": normalized or g.get("priority"),
                "futureCostInflated": g.get("futureCostInflated"),
                "assumedReturnAnnual": g.get("assumedReturnAnnual"),
                "requiredMonthlyInvestmentRounded": g.get("requiredMonthlyInvestmentRounded"),
                "shortfallMonthlyRounded": g.get("shortfallMonthlyRounded"),
                "status": g.get("status"),
                "_rank": rank,
                "_horizon": horizon_value if horizon_value is not None else 1000000000.0,
            }
        )
    goal_items.sort(key=lambda item: (item.get("_rank", 3), item.get("_horizon", 0.0)))
    top_goals = [
        {k: v for k, v in g.items() if not k.startswith("_")}
        for g in goal_items[:3]
    ]

    liabilities = (inputs.get("liabilities") or {}).get("outstanding") or {}
    emis = (inputs.get("liabilities") or {}).get("emi") or {}
    remaining = (inputs.get("liabilities") or {}).get("remainingMonths") or {}
    top_liabilities: List[Dict[str, Any]] = []
    for key, value in liabilities.items():
        if str(key) == "totalOutstanding":
            continue
        outstanding = _safe_float(value)
        if outstanding is None or outstanding <= 0:
            continue
        top_liabilities.append(
            {
                "key": key,
                "label": _humanize_key(str(key)),
                "outstanding": value,
                "emi": emis.get(f"{key}Emi") if isinstance(emis, dict) else None,
                "remainingMonths": remaining.get(f"{key}RemainingMonths")
                if isinstance(remaining, dict)
                else None,
            }
        )

    return {
        "monthlyIncomeTotal": _get_path(derived, ["income", "monthlyTotal"]),
        "monthlyExpenseExclEmi": _get_path(derived, ["outflow", "monthlyExpenseExclEmi"]),
        "monthlyEmiTotal": _get_path(derived, ["outflow", "monthlyEmiTotal"]),
        "monthlySurplus": _get_path(derived, ["surplus", "monthlySurplus"]),
        "emergency": _get_path(derived, ["emergency"], {}),
        "topGoals": top_goals,
        "topLiabilities": top_liabilities,
        "assumptions": _get_path(blueprint, ["meta", "assumptions"])
        or blueprint.get("assumptions")
        or {},
        "warnings": blueprint.get("warnings") or [],
    }


def _deterministic_health_label(blueprint: Dict[str, Any]) -> str:
    monthly_surplus = _safe_float(_get_path(blueprint, ["derived", "surplus", "monthlySurplus"]))
    emergency_gap = _safe_float(_get_path(blueprint, ["derived", "emergency", "gap"]))

    if monthly_surplus is not None and monthly_surplus <= 0:
        return "At Risk"
    if emergency_gap is not None and emergency_gap > 0:
        return "Needs Attention"
    return "Healthy"


def _deterministic_risk_band(blueprint: Dict[str, Any], plan: Dict[str, Any]) -> str:
    quiz_label = _get_path(blueprint, ["derived", "risk", "riskQuizLabel"])
    if quiz_label:
        return str(quiz_label)
    plan_risk = (plan.get("risk") or {}).get("finalBand") or (plan.get("risk") or {}).get("capacityBand")
    return str(plan_risk or "Balanced")


def _build_checklist_seeds(
    blueprint: Dict[str, Any], facts: Dict[str, Any], plan: Dict[str, Any]
) -> List[Dict[str, Any]]:
    seeds: List[Dict[str, Any]] = []

    monthly_surplus = _safe_float(facts.get("monthlySurplus"))
    monthly_income = _safe_float(facts.get("monthlyIncomeTotal"))
    monthly_emi = _safe_float(facts.get("monthlyEmiTotal"))
    emergency = facts.get("emergency") or {}
    emergency_gap = _safe_float(emergency.get("gap"))
    emergency_status = emergency.get("status")
    contributions_sum = _safe_float(
        _get_path(blueprint, ["inputsSnapshot", "contributions", "sumMonthly"])
    )
    goals_total = len(_get_path(blueprint, ["derived", "goals"], []) or [])

    if monthly_surplus is not None and monthly_surplus <= 0:
        seeds.append(
            {
                "id": "stabilize-surplus",
                "title": "Restore positive monthly surplus",
                "why": "Monthly surplus is not positive in the blueprint.",
                "priority": "High",
            }
        )

    if emergency_status == "Underfunded" and emergency_gap is not None and emergency_gap > 0:
        seeds.append(
            {
                "id": "build-emergency",
                "title": f"Build emergency fund gap of {format_inr(emergency_gap)}",
                "why": "Emergency buffer is underfunded based on target corpus vs current dedicated amount.",
                "priority": "High",
            }
        )

    liabilities = _get_path(blueprint, ["inputsSnapshot", "liabilities", "outstanding"], {}) or {}
    has_credit_card = False
    for key, value in liabilities.items():
        if "creditcard" in str(key).lower():
            v = _safe_float(value)
            if v is not None and v > 0:
                has_credit_card = True
                break
    if has_credit_card:
        seeds.append(
            {
                "id": "pay-credit-card",
                "title": "Pay down credit card balances",
                "why": "Credit card outstanding appears in liabilities.",
                "priority": "High",
            }
        )

    if monthly_income and monthly_income > 0 and monthly_emi is not None:
        if (monthly_emi / monthly_income) >= 0.4:
            seeds.append(
                {
                    "id": "review-emi-burden",
                    "title": "Review refinancing or prepayment options",
                    "why": "Monthly EMI is a large share of income in the blueprint.",
                    "priority": "Medium",
                }
            )

    if contributions_sum is not None and contributions_sum <= 0 and monthly_surplus and monthly_surplus > 0:
        seeds.append(
            {
                "id": "start-sip",
                "title": "Start SIPs aligned to target allocation",
                "why": "There is positive surplus but no monthly contributions recorded.",
                "priority": "Medium",
            }
        )

    if len(seeds) < 3:
        seeds.append(
            {
                "id": "review-plan",
                "title": "Review recommended plan selection",
                "why": "Confirm the chosen risk band and allocation direction.",
                "priority": "Medium",
            }
        )

    if len(seeds) < 3:
        if goals_total < 2:
            seeds.append(
                {
                    "id": "add-goals",
                    "title": "Add missing goal details",
                    "why": "More goal details improve the timeline and funding order.",
                    "priority": "Low",
                }
            )
        else:
            seeds.append(
                {
                    "id": "confirm-liabilities",
                    "title": "Confirm liability details",
                    "why": "Accurate outstanding/EMI values keep the cash flow view realistic.",
                    "priority": "Low",
                }
            )

    if len(seeds) < 3:
        seeds.append(
            {
                "id": "confirm-inputs",
                "title": "Confirm cash flow inputs",
                "why": "Expense and EMI inputs drive the surplus and emergency buffer calculations.",
                "priority": "Low",
            }
        )

    if len(seeds) > 5:
        seeds = seeds[:5]

    return seeds


def _build_emergency_text(emergency: Dict[str, Any], use_format: bool) -> str:
    target_months = _safe_number_text(emergency.get("targetMonths"))
    status = emergency.get("status") or "Unknown"

    if use_format:
        target_corpus = format_inr(emergency.get("targetCorpus"))
        gap = format_inr(emergency.get("gap"))
        surplus = format_inr(emergency.get("surplus"))
    else:
        target_corpus = _safe_number_text(emergency.get("targetCorpus"))
        gap = _safe_number_text(emergency.get("gap"))
        surplus = _safe_number_text(emergency.get("surplus"))

    line = (
        f"Target months: {target_months}. "
        f"Target corpus: {target_corpus}. "
        f"Gap: {gap}. Surplus: {surplus}. Status: {status}."
    )

    if status == "Underfunded":
        line += f" You are short by {gap} toward your emergency fund target."
    elif status == "Adequate":
        surplus_value = _safe_float(emergency.get("surplus")) or 0.0
        if surplus_value > 0:
            line += f" You are above target by {surplus} in your emergency fund."
        else:
            line += " You are exactly at your emergency fund target."
    else:
        line += " Emergency status is unknown; please confirm your target months and monthly expense."

    return line


def _replace_value_tokens(text: str, values: List[Any]) -> str:
    if not text:
        return text
    updated = text
    for value in values:
        if value is None:
            continue
        formatted = format_inr(value)
        variants = set(_numeric_variants(value))
        formatted_no_symbol = formatted.replace("\u20B9", "")
        if formatted_no_symbol:
            variants.add(formatted_no_symbol)
        for variant in variants:
            pattern = rf"(?<![0-9])\u20B9?{re.escape(variant)}(?![0-9])"
            updated = re.sub(pattern, formatted, updated)
    return updated


def _rewrite_emergency_section(
    response: NarrationResponse, facts: Dict[str, Any], use_format: bool
) -> NarrationResponse:
    emergency = facts.get("emergency") or {}
    for section in response.sections:
        if "emergency buffer" in (section.title or "").lower():
            section.markdown = _build_emergency_text(emergency, use_format)
            break
    return response


def _apply_inr_formatting(response: NarrationResponse, facts: Dict[str, Any]) -> NarrationResponse:
    cash_values = [
        facts.get("monthlyExpenseExclEmi"),
        facts.get("monthlyEmiTotal"),
        facts.get("monthlySurplus"),
    ]
    emergency = facts.get("emergency") or {}
    emergency_values = [
        emergency.get("targetCorpus"),
        emergency.get("gap"),
        emergency.get("surplus"),
    ]
    goal_values: List[Any] = []
    for goal in facts.get("topGoals") or []:
        goal_values.append((goal or {}).get("amountToday"))
        goal_values.append((goal or {}).get("futureCostInflated"))
        goal_values.append((goal or {}).get("requiredMonthlyInvestmentRounded"))
        goal_values.append((goal or {}).get("shortfallMonthlyRounded"))
    liability_values: List[Any] = []
    for item in facts.get("topLiabilities") or []:
        liability_values.append(item.get("outstanding"))
        liability_values.append(item.get("emi"))

    for section in response.sections:
        title = (section.title or "").lower()
        if "cash flow snapshot" in title:
            section.markdown = _replace_value_tokens(section.markdown, cash_values)
        elif "goals & timeline" in title:
            section.markdown = _replace_value_tokens(section.markdown, goal_values)
        elif "debt & next 30 days" in title:
            section.markdown = _replace_value_tokens(section.markdown, liability_values)

    response.planSummary.oneLiner = _replace_value_tokens(
        response.planSummary.oneLiner, cash_values + emergency_values
    )

    all_values = cash_values + emergency_values + goal_values + liability_values
    for item in response.actionChecklist:
        item.title = _replace_value_tokens(item.title, all_values)
        item.why = _replace_value_tokens(item.why, all_values)

    return response


def _fallback_narration(blueprint: Dict[str, Any], plan: Dict[str, Any]) -> NarrationResponse:
    warnings = blueprint.get("warnings") or []
    clarifying = _warnings_to_questions(warnings) if warnings else []
    facts = _build_facts(blueprint)

    monthly_expense = facts.get("monthlyExpenseExclEmi")
    monthly_emi = facts.get("monthlyEmiTotal")
    monthly_surplus = facts.get("monthlySurplus")
    emergency = facts.get("emergency") or {}
    emergency_corpus = emergency.get("targetCorpus")
    emergency_months = emergency.get("targetMonths")

    monthly_expense_fmt = format_inr(monthly_expense)
    monthly_emi_fmt = format_inr(monthly_emi)
    monthly_surplus_fmt = format_inr(monthly_surplus)
    emergency_corpus_fmt = format_inr(emergency_corpus)

    summary = PlanSummary(
        oneLiner=(
            "Cash flow and emergency buffer are described using blueprint values: "
            f"expense={monthly_expense_fmt}, EMI={monthly_emi_fmt}, surplus={monthly_surplus_fmt}."
        ),
        healthScoreLabel=_deterministic_health_label(blueprint),
        riskBand=_deterministic_risk_band(blueprint, plan),
    )

    emergency_line = _build_emergency_text(emergency, use_format=True)

    goal_lines = []
    for goal in facts.get("topGoals") or []:
        amount_fmt = format_inr(goal.get("amountToday"))
        goal_lines.append(
            f"{goal.get('label') or goal.get('key')}: amount={amount_fmt}, "
            f"horizon={goal.get('horizonYears')}, priority={goal.get('priority')}."
        )
    goals_markdown = (
        " ".join(goal_lines)
        if goal_lines
        else "Goals are missing from the blueprint. Please share at least two goals with amount, horizon years, and priority."
    )

    liability_lines = []
    for liab in facts.get("topLiabilities") or []:
        outstanding_fmt = format_inr(liab.get("outstanding"))
        emi_fmt = format_inr(liab.get("emi"))
        line = (
            f"{liab.get('label') or liab.get('key')}: "
            f"outstanding={outstanding_fmt}, EMI={emi_fmt}."
        )
        if liab.get("remainingMonths") is not None:
            line += f" Remaining months={liab.get('remainingMonths')}."
        liability_lines.append(line)
    liabilities_markdown = (
        " ".join(liability_lines)
        if liability_lines
        else "Liability details are missing. Do you have any active loans or EMIs? Please share outstanding, EMI, and remaining months."
    )

    sections = [
        NarrationSection(
            title="Cash flow snapshot",
            markdown=(
                f"Monthly expense (excl. EMI) is {monthly_expense_fmt}, "
                f"monthly EMI total is {monthly_emi_fmt}, and monthly surplus is {monthly_surplus_fmt}."
            ),
        ),
        NarrationSection(
            title="Emergency buffer",
            markdown=emergency_line,
        ),
        NarrationSection(
            title="Goals & timeline",
            markdown=goals_markdown,
        ),
        NarrationSection(
            title="Debt & next 30 days",
            markdown=liabilities_markdown,
        ),
    ]

    checklist_seeds = _build_checklist_seeds(blueprint, facts, plan)
    action_items = [
        ActionChecklistItem(
            id=seed["id"],
            title=seed["title"],
            why=seed["why"],
            priority=seed["priority"],
        )
        for seed in checklist_seeds
    ]

    disclosures = [
        "This report explains the stored blueprint and plan without calculating new figures.",
        "Any missing data is noted as a clarification request.",
    ]

    response = NarrationResponse(
        narrationVersion=NARRATION_VERSION,
        generatedAt=_now_iso(),
        planSummary=summary,
        sections=sections,
        actionChecklist=action_items,
        clarifyingQuestions=clarifying,
        disclosures=disclosures,
    )
    response = _ensure_required_clarifications(response, blueprint, facts)
    return response


def _normalize_number(token: str) -> str:
    return _NON_NUMERIC_RE.sub("", token or "")


def _numeric_variants(value: float | int) -> Iterable[str]:
    if isinstance(value, bool):
        return []
    if isinstance(value, int):
        return [str(value)]
    if isinstance(value, float):
        tokens = {format(value, "g"), str(value)}
        if value.is_integer():
            tokens.add(str(int(value)))
        return tokens
    return []


def _collect_allowed_numbers(data: Any) -> set[str]:
    allowed: set[str] = set()

    def walk(val: Any) -> None:
        if isinstance(val, dict):
            for v in val.values():
                walk(v)
        elif isinstance(val, list):
            for item in val:
                walk(item)
        elif isinstance(val, (int, float)) and not isinstance(val, bool):
            for token in _numeric_variants(val):
                norm = _normalize_number(token)
                if norm:
                    allowed.add(norm)
        elif isinstance(val, str):
            for match in _NUMBER_RE.finditer(val):
                norm = _normalize_number(match.group(0))
                if norm:
                    allowed.add(norm)

    walk(data)
    return allowed


def _collect_response_numbers(data: Any) -> set[str]:
    numbers: set[str] = set()

    def walk(val: Any) -> None:
        if isinstance(val, dict):
            for v in val.values():
                walk(v)
        elif isinstance(val, list):
            for item in val:
                walk(item)
        elif isinstance(val, (int, float)) and not isinstance(val, bool):
            for token in _numeric_variants(val):
                norm = _normalize_number(token)
                if norm:
                    numbers.add(norm)
        elif isinstance(val, str):
            for match in _NUMBER_RE.finditer(val):
                norm = _normalize_number(match.group(0))
                if norm:
                    numbers.add(norm)

    walk(data)
    return numbers


def _numbers_allowed(response: Dict[str, Any], blueprint: Dict[str, Any], plan: Dict[str, Any]) -> bool:
    allowed = _collect_allowed_numbers({"blueprint": blueprint, "plan": plan})
    if not allowed:
        allowed = set()
    scrubbed = dict(response)
    scrubbed.pop("generatedAt", None)
    scrubbed.pop("narrationVersion", None)
    response_numbers = _collect_response_numbers(scrubbed)
    return response_numbers.issubset(allowed)


def _contains_number(text: str, number_str: str) -> bool:
    if not number_str:
        return False
    pattern = rf"(?<!\d){re.escape(number_str)}(?!\d)"
    return re.search(pattern, text) is not None


def _ensure_warning_questions(
    response: NarrationResponse, blueprint: Dict[str, Any]
) -> NarrationResponse:
    warnings = blueprint.get("warnings") or []
    if not warnings:
        return response
    if response.clarifyingQuestions:
        return response

    patched = response.dict()
    patched["clarifyingQuestions"] = [q.dict() for q in _warnings_to_questions(warnings)]
    return NarrationResponse.parse_obj(patched)


def _ensure_required_clarifications(
    response: NarrationResponse, blueprint: Dict[str, Any], facts: Dict[str, Any]
) -> NarrationResponse:
    response = _ensure_warning_questions(response, blueprint)
    clarifying = [q.dict() for q in response.clarifyingQuestions]
    existing_ids = {q["id"] for q in clarifying if isinstance(q, dict)}

    goals_total = len(_get_path(blueprint, ["derived", "goals"], []) or [])
    if goals_total < 2 and "clarify-goals" not in existing_ids:
        if goals_total == 0:
            question = "Please share at least two financial goals with amount, horizon years, and priority."
        else:
            question = "Please share at least one more financial goal with amount, horizon years, and priority."
        clarifying.append(
            {
                "id": "clarify-goals",
                "question": question,
                "whyItMatters": "Goal details are required to build the timeline and funding order.",
            }
        )

    if not (facts.get("topLiabilities") or []) and "clarify-liabilities" not in existing_ids:
        clarifying.append(
            {
                "id": "clarify-liabilities",
                "question": "Do you have any active loans or EMIs? Please share outstanding, EMI, and remaining months.",
                "whyItMatters": "Debt details affect cash flow and the next-30-days action plan.",
            }
        )

    emergency_status = (facts.get("emergency") or {}).get("status")
    if emergency_status == "Unknown" and "clarify-emergency" not in existing_ids:
        clarifying.append(
            {
                "id": "clarify-emergency",
                "question": "Please confirm your emergency fund target months and total monthly expense.",
                "whyItMatters": "These inputs determine your emergency fund target and gap.",
            }
        )

    patched = response.dict()
    patched["clarifyingQuestions"] = clarifying
    return NarrationResponse.parse_obj(patched)


def _apply_deterministic_plan_summary(
    response: NarrationResponse, blueprint: Dict[str, Any], plan: Dict[str, Any]
) -> NarrationResponse:
    response.planSummary.healthScoreLabel = _deterministic_health_label(blueprint)
    response.planSummary.riskBand = _deterministic_risk_band(blueprint, plan)
    return response


def _validate_response(
    data: Dict[str, Any], blueprint: Dict[str, Any], plan: Dict[str, Any]
) -> Optional[NarrationResponse]:
    try:
        parsed = NarrationResponse.parse_obj(data)
    except ValidationError:
        return None
    if not _numbers_allowed(data, blueprint, plan):
        return None
    return parsed


def _required_number_values(blueprint: Dict[str, Any]) -> Dict[str, Optional[float]]:
    return {
        "monthlyEmiTotal": _safe_float(_get_path(blueprint, ["derived", "outflow", "monthlyEmiTotal"])),
        "monthlyExpenseExclEmi": _safe_float(
            _get_path(blueprint, ["derived", "outflow", "monthlyExpenseExclEmi"])
        ),
        "monthlySurplus": _safe_float(_get_path(blueprint, ["derived", "surplus", "monthlySurplus"])),
        "emergencyTargetMonths": _safe_float(_get_path(blueprint, ["derived", "emergency", "targetMonths"])),
        "emergencyTargetCorpus": _safe_float(_get_path(blueprint, ["derived", "emergency", "targetCorpus"])),
        "emergencyGap": _safe_float(_get_path(blueprint, ["derived", "emergency", "gap"])),
    }


def _number_string_variants(value: Optional[float]) -> List[str]:
    if value is None:
        return []
    variants = list(_numeric_variants(value))
    return [v for v in variants if v]


def _quality_check(
    response: NarrationResponse, facts: Dict[str, Any], required_numbers: Dict[str, Optional[float]]
) -> Tuple[bool, Dict[str, Any]]:
    sections_text = "\n".join(section.markdown or "" for section in response.sections)
    sections_lower = sections_text.lower()

    numeric_tokens = _NUMBER_RE.findall(sections_text)
    numeric_count_ok = len(numeric_tokens) >= 6

    sections_count_ok = len(response.sections) >= 4
    required_titles = [
        "cash flow snapshot",
        "emergency buffer",
        "goals & timeline",
        "debt & next 30 days",
    ]
    section_titles = [str(section.title or "").strip().lower() for section in response.sections]
    missing_sections = [
        title for title in required_titles if not any(title in candidate for candidate in section_titles)
    ]
    section_titles_ok = len(missing_sections) == 0

    missing_required_numbers: List[str] = []
    required_number_hits = 0
    required_number_expected = 0
    for key, value in required_numbers.items():
        if value is None:
            missing_required_numbers.append(key)
            continue
        required_number_expected += 1
        variants = _number_string_variants(value)
        if any(_contains_number(sections_text, v) for v in variants):
            required_number_hits += 1
        else:
            missing_required_numbers.append(key)

    required_numbers_ok = (
        True if required_number_expected == 0 else required_number_hits >= required_number_expected
    )

    emergency = facts.get("emergency") or {}
    emergency_status = str(emergency.get("status") or "").lower()
    emergency_status_ok = not emergency_status or emergency_status in sections_lower

    emergency_phrase_ok = True
    emergency_gap = _safe_float(emergency.get("gap"))
    emergency_surplus = _safe_float(emergency.get("surplus"))
    if emergency_status == "underfunded" and emergency_gap is not None and emergency_gap > 0:
        emergency_phrase_ok = (
            "short by" in sections_lower
            and any(_contains_number(sections_text, v) for v in _number_string_variants(emergency_gap))
        )
    if emergency_status == "adequate" and emergency_surplus is not None and emergency_surplus > 0:
        emergency_phrase_ok = (
            "above target by" in sections_lower
            and any(_contains_number(sections_text, v) for v in _number_string_variants(emergency_surplus))
        )

    top_liabilities = facts.get("topLiabilities") or []
    liability_matches = 0
    for item in top_liabilities:
        key = str(item.get("key") or "").lower()
        label = str(item.get("label") or "").lower()
        if key and key in sections_lower:
            liability_matches += 1
            break
        if label and label in sections_lower:
            liability_matches += 1
            break
    liabilities_ok = not top_liabilities or liability_matches >= 1
    missing_liabilities = [item.get("key") for item in top_liabilities] if not liabilities_ok else []

    top_goals = facts.get("topGoals") or []
    goal_matches = 0
    missing_goals: List[Any] = []
    for item in top_goals:
        key = str(item.get("key") or "").lower()
        label = str(item.get("label") or "").lower()
        matched = False
        if key and key in sections_lower:
            matched = True
        if label and label in sections_lower:
            matched = True
        if matched:
            goal_matches += 1
        else:
            missing_goals.append(item.get("key") or item.get("label"))
    goals_required = 2 if len(top_goals) >= 2 else (1 if len(top_goals) == 1 else 0)
    goals_ok = goal_matches >= goals_required

    checklist_count = len(response.actionChecklist or [])
    checklist_ok = 3 <= checklist_count <= 5

    ok = (
        numeric_count_ok
        and required_numbers_ok
        and liabilities_ok
        and goals_ok
        and sections_count_ok
        and section_titles_ok
        and checklist_ok
        and emergency_status_ok
        and emergency_phrase_ok
    )
    return ok, {
        "numericCountOk": numeric_count_ok,
        "requiredNumbersOk": required_numbers_ok,
        "missingRequiredNumbers": missing_required_numbers,
        "missingLiabilities": missing_liabilities,
        "missingGoals": missing_goals,
        "requiredNumbersHit": required_number_hits,
        "requiredNumbersExpected": required_number_expected,
        "numericCount": len(numeric_tokens),
        "sectionsCountOk": sections_count_ok,
        "sectionTitlesOk": section_titles_ok,
        "missingSections": missing_sections,
        "checklistOk": checklist_ok,
        "checklistCount": checklist_count,
        "emergencyStatusOk": emergency_status_ok,
        "emergencyPhraseOk": emergency_phrase_ok,
    }


def _revision_prompt(issues: Dict[str, Any], required_numbers: Dict[str, Optional[float]]) -> str:
    missing_numbers = issues.get("missingRequiredNumbers") or []
    missing_liabilities = issues.get("missingLiabilities") or []
    missing_goals = issues.get("missingGoals") or []
    missing_sections = issues.get("missingSections") or []
    checklist_count = issues.get("checklistCount")

    required_values: Dict[str, Any] = {}
    for key, value in required_numbers.items():
        if value is not None:
            required_values[key] = value

    return (
        "REVISION REQUIRED.\n"
        "Fix the output to be grounded and specific. Remove generic filler.\n"
        f"Missing required numbers (include these exact values if present in data): {required_values}\n"
        f"Missing required numbers by key: {missing_numbers}\n"
        f"Missing required sections by title: {missing_sections}\n"
        f"Checklist count (must be 3-5): {checklist_count}\n"
        f"Liabilities to mention (key/label): {missing_liabilities}\n"
        f"Goals to mention (key/label): {missing_goals}\n"
        "Remember: include at least 6 numeric facts and cover the required emergency and cashflow values.\n"
        "Return ONLY NarrationResponse JSON."
    )


def generate_narration(blueprint: Dict[str, Any], plan: Dict[str, Any]) -> NarrationResponse:
    model = os.getenv("LLM_MODEL_NARRATOR", DEFAULT_MODEL)
    temperature = _env_float("LLM_TEMPERATURE_NARRATOR", DEFAULT_TEMPERATURE)

    facts = _build_facts(blueprint)
    checklist_seeds = _build_checklist_seeds(blueprint, facts, plan)
    system = _system_prompt()
    user = _user_prompt(blueprint, plan, facts, checklist_seeds)
    required_numbers = _required_number_values(blueprint)

    raw = _call_openai(system, user, model, temperature)
    data = _parse_json(raw)
    parsed = _validate_response(data or {}, blueprint, plan) if data else None
    if not parsed:
        fix_prompt = (
            "The previous response was invalid JSON or violated the number constraints. "
            "Return ONLY valid NarrationResponse JSON with no extra text."
        )
        raw = _call_openai(system, f"{user}\n\n{fix_prompt}\n\nPrevious response:\n{raw}", model, temperature)
        data = _parse_json(raw)
        parsed = _validate_response(data or {}, blueprint, plan) if data else None

    if parsed:
        parsed = _apply_deterministic_plan_summary(parsed, blueprint, plan)
        parsed = _ensure_required_clarifications(parsed, blueprint, facts)
        parsed = _rewrite_emergency_section(parsed, facts, use_format=False)
        ok, issues = _quality_check(parsed, facts, required_numbers)
        if ok:
            parsed = _rewrite_emergency_section(parsed, facts, use_format=True)
            parsed = _apply_inr_formatting(parsed, facts)
            return parsed

        revision = _revision_prompt(issues, required_numbers)
        raw = _call_openai(system, f"{user}\n\n{revision}", model, temperature)
        data = _parse_json(raw)
        parsed = _validate_response(data or {}, blueprint, plan) if data else None
        if parsed:
            parsed = _apply_deterministic_plan_summary(parsed, blueprint, plan)
            parsed = _ensure_required_clarifications(parsed, blueprint, facts)
            parsed = _rewrite_emergency_section(parsed, facts, use_format=False)
            ok, _ = _quality_check(parsed, facts, required_numbers)
            if ok:
                parsed = _rewrite_emergency_section(parsed, facts, use_format=True)
                parsed = _apply_inr_formatting(parsed, facts)
                return parsed

    return _fallback_narration(blueprint, plan)
