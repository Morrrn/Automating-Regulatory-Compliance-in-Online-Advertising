from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

# define the values allowed by the annotation schema
PRIMARY_CODE_AREAS = {
    "misleading_advertising",
    "recognition_of_advertising",
    "harm_or_offence",
    "health_medical_or_wellbeing_claim",
    "financial_or_business_opportunity",
    "other_or_unclear",
}

EVIDENCE_TYPES = {
    "objective_claim_requires_substantiation",
    "missing_material_information",
    "unclear_or_hidden_commercial_intent",
    "exaggerated_capability_or_performance",
    "misleading_price_or_offer",
    "fear_distress_or_shock_tactic",
    "offensive_or_harmful_stereotype",
    "health_cure_prevention_or_weight_loss_claim",
    "financial_return_or_investment_risk_claim",
    "earnings_or_business_opportunity_claim",
    "cross_modal_mismatch",
    "insufficient_information",
}

ALLOWED_RULE_REFS = {
    "1.1",
    "1.2",
    "1.3",
    "2.1",
    "2.3",
    "2.4",
    "3.1",
    "3.3",
    "3.5",
    "3.6",
    "3.7",
    "3.9",
    "3.10",
    "3.11",
    "3.17",
    "4.1",
    "4.2",
    "4.4",
    "4.5",
    "4.9",
    "12",
    "13.8",
    "13.9",
    "15.1",
    "15.6",
    "14.1",
    "14.2",
    "14.4",
    "14.5",
    "20.1",
    "20.2",
    "20.7",
    "20.8",
}

# list every field required in a model response
REQUIRED_FIELDS = {
    "is_potential_violation",
    "risk_score",
    "primary_code_area",
    "secondary_code_areas",
    "evidence_type",
    "cap_code_rule_refs",
    "ad_text_summary",
    "visual_summary",
    "rationale",
    "confidence",
    "needs_human_review",
}

CONFIDENCE_VALUES = {"low", "medium", "high"}

# store parsing and schema validation results
@dataclass
class ValidationResult:
    parse_ok: bool
    schema_ok: bool
    errors: list[str]
    parsed: dict[str, Any] | None = None

# Extract a JSON object from plain text or fenced markdown.
def extract_json_object(raw: str) -> str:
    # remove markdown fences before locating the json object
    raw = (raw or "").strip()
    if raw.startswith("```"):
        lines = raw.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        raw = "\n".join(lines).strip()
    start = raw.find("{")
    end = raw.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return raw
    return raw[start : end + 1]


def validate_annotation(raw: str) -> ValidationResult:
    # keep parsing errors separate from schema errors
    errors: list[str] = []
    try:
        parsed = json.loads(extract_json_object(raw))
    except Exception as exc:  # noqa: BLE001 - report parser failure clearly for research log
        return ValidationResult(parse_ok=False, schema_ok=False, errors=[f"json_parse_error: {exc}"], parsed=None)

    if not isinstance(parsed, dict):
        return ValidationResult(parse_ok=True, schema_ok=False, errors=["top_level_json_is_not_object"], parsed=None)

    # compare response fields with the shared schema
    missing = sorted(REQUIRED_FIELDS - set(parsed))
    extra = sorted(set(parsed) - REQUIRED_FIELDS)
    if missing:
        errors.append("missing_fields: " + ",".join(missing))
    if extra:
        errors.append("extra_fields: " + ",".join(extra))

    # validate the decision and risk score
    if parsed.get("is_potential_violation") not in (0, 1):
        errors.append("is_potential_violation_must_be_0_or_1")

    risk_score = parsed.get("risk_score")
    if not isinstance(risk_score, (int, float)) or not 0 <= float(risk_score) <= 1:
        errors.append("risk_score_must_be_number_between_0_and_1")

    # validate primary and secondary area labels
    if parsed.get("primary_code_area") not in PRIMARY_CODE_AREAS:
        errors.append("invalid_primary_code_area")

    secondary = parsed.get("secondary_code_areas")
    if not isinstance(secondary, list):
        errors.append("secondary_code_areas_must_be_list")
    else:
        invalid = sorted(set(secondary) - PRIMARY_CODE_AREAS)
        if invalid:
            errors.append("invalid_secondary_code_areas: " + ",".join(invalid))

    # validate evidence labels
    evidence = parsed.get("evidence_type")
    if not isinstance(evidence, list):
        errors.append("evidence_type_must_be_list")
    else:
        invalid = sorted(set(evidence) - EVIDENCE_TYPES)
        if invalid:
            errors.append("invalid_evidence_type: " + ",".join(invalid))

    # validate cap code references
    refs = parsed.get("cap_code_rule_refs")
    if not isinstance(refs, list):
        errors.append("cap_code_rule_refs_must_be_list")
    else:
        string_refs = [ref for ref in refs if isinstance(ref, str)]
        if len(string_refs) != len(refs):
            errors.append("cap_code_rule_refs_must_contain_only_strings")
        invalid = sorted(set(string_refs) - ALLOWED_RULE_REFS)
        if invalid:
            errors.append("invalid_cap_code_rule_refs: " + ",".join(invalid))

    # validate confidence review flag and text fields
    if parsed.get("confidence") not in CONFIDENCE_VALUES:
        errors.append("confidence_must_be_low_medium_or_high")

    if not isinstance(parsed.get("needs_human_review"), bool):
        errors.append("needs_human_review_must_be_boolean")

    for field in ["ad_text_summary", "visual_summary", "rationale"]:
        if not isinstance(parsed.get(field), str):
            errors.append(f"{field}_must_be_string")

    return ValidationResult(parse_ok=True, schema_ok=not errors, errors=errors, parsed=parsed)
