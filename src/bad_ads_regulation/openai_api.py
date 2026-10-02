from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Any

from bad_ads_regulation.annotation import (
    ALLOWED_RULE_REFS,
    CONFIDENCE_VALUES,
    EVIDENCE_TYPES,
    PRIMARY_CODE_AREAS,
)


def annotation_json_schema() -> dict[str, Any]:
    # build the strict schema from the shared annotation values
    return {
        "type": "object",
        "additionalProperties": False,
        "required": [
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
        ],
        "properties": {
            "is_potential_violation": {"type": "integer", "enum": [0, 1]},
            "risk_score": {"type": "number", "minimum": 0, "maximum": 1},
            "primary_code_area": {"type": "string", "enum": sorted(PRIMARY_CODE_AREAS)},
            "secondary_code_areas": {
                "type": "array",
                "items": {"type": "string", "enum": sorted(PRIMARY_CODE_AREAS)},
            },
            "evidence_type": {
                "type": "array",
                "items": {"type": "string", "enum": sorted(EVIDENCE_TYPES)},
            },
            "cap_code_rule_refs": {
                "type": "array",
                "items": {"type": "string", "enum": sorted(ALLOWED_RULE_REFS)},
            },
            "ad_text_summary": {"type": "string"},
            "visual_summary": {"type": "string"},
            "rationale": {"type": "string"},
            "confidence": {"type": "string", "enum": sorted(CONFIDENCE_VALUES)},
            "needs_human_review": {"type": "boolean"},
        },
    }


def image_to_data_url(path: str | Path) -> str:
    # choose the image type and encode the file as a data url
    path = Path(path)
    suffix = path.suffix.lower()
    mime = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".webp": "image/webp",
    }.get(suffix, "application/octet-stream")
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{data}"


def build_user_text(prompt: str, ad_id: int | str, ocr_text: str = "") -> str:
    # add the ad id and available ocr text to the prompt
    ocr = ocr_text.strip() if ocr_text else "not provided. Please read visible text from the image directly."
    return f"{prompt.strip()}\n\nAd ID: {ad_id}\nOCR text: {ocr}"


def build_responses_payload(
    *,
    prompt: str,
    ad_id: int | str,
    image_path: str | Path,
    model: str,
    ocr_text: str = "",
    image_detail: str = "low",
    max_output_tokens: int = 900,
) -> dict[str, Any]:
    # combine text image and schema settings into one request
    return {
        "model": model,
        "input": [
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": build_user_text(prompt, ad_id, ocr_text)},
                    {"type": "input_image", "image_url": image_to_data_url(image_path), "detail": image_detail},
                ],
            }
        ],
        "text": {
            "format": {
                "type": "json_schema",
                "name": "cap_ad_annotation",
                "strict": True,
                "schema": annotation_json_schema(),
            }
        },
        "max_output_tokens": max_output_tokens,
    }


def preview_payload(payload: dict[str, Any]) -> dict[str, Any]:
    # copy the payload before hiding the large image data
    preview = json.loads(json.dumps(payload))
    content = preview["input"][0]["content"]
    for item in content:
        if item.get("type") == "input_image":
            image_url = item.get("image_url", "")
            item["image_url"] = f"<data-url omitted; {len(image_url)} chars>"
    return preview


def extract_output_text(response: dict[str, Any]) -> str:
    # use the direct text field before searching nested output items
    if isinstance(response.get("output_text"), str):
        return response["output_text"]
    parts: list[str] = []
    for item in response.get("output", []) or []:
        for content in item.get("content", []) or []:
            if content.get("type") in {"output_text", "text"} and isinstance(content.get("text"), str):
                parts.append(content["text"])
    return "\n".join(parts).strip()

