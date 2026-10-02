from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
# group perception labels into negative and positive sets
NEGATIVE_LABELS = [
    "bad_design",
    "clickbait",
    "deceptive",
    "distasteful",
    "manipulative",
    "unclear",
]

POSITIVE_LABELS = ["entertaining", "good_design", "like_product", "simple", "trustworthy", "useful"]

# map content keywords to broad risk domains
RISK_CONTENT_KEYWORDS = {
    "health": ["health", "medical", "medicine", "beauty", "weight", "fitness", "supplement", "diet"],
    "finance": ["finance", "financial", "insurance", "loan", "credit", "investment", "crypto", "business"],
    "employment": ["job", "employment", "career", "work", "business opportunity", "homework"],
}


def load_ads(dataset_root: str | Path) -> list[dict[str, Any]]:
    # load the source json and confirm it contains a list of ads
    dataset_root = Path(dataset_root)
    labels_path = dataset_root / "data" / "ads_all_labels.json"
    with labels_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("ads_all_labels.json should contain a list of ads")
    return data


def resolve_screenshot(dataset_root: str | Path, ad: dict[str, Any]) -> Path:
    # use the recorded filename before trying an id-based webp name
    dataset_root = Path(dataset_root)
    screenshot = str(ad.get("screenshot", ""))
    direct = dataset_root / "screenshots" / screenshot
    if direct.exists():
        return direct
    ad_id = str(ad.get("id"))
    webp = dataset_root / "screenshots" / f"{ad_id}.webp"
    if webp.exists():
        return webp
    return direct


def _sum_labels(label_dist: dict[str, float], labels: list[str]) -> float:
    # total the selected perception label scores
    return float(sum(float(label_dist.get(label, 0.0)) for label in labels))


def _risk_domain(content_labels: list[str]) -> str:
    # match content labels against simple risk keywords
    joined = " ".join(content_labels).lower()
    matched = [domain for domain, keys in RISK_CONTENT_KEYWORDS.items() if any(key in joined for key in keys)]
    return ";".join(matched) if matched else ""


def build_inventory(dataset_root: str | Path) -> pd.DataFrame:
    # turn the source ads into one flat inventory row each
    ads = load_ads(dataset_root)
    rows: list[dict[str, Any]] = []
    # collect source fields and locate the matching screenshot
    for ad in ads:
        label_dist = ad.get("opinion_label_dist", {}) or {}
        content_labels = ad.get("content_labels", []) or []
        ratings = ad.get("ratings", []) or []
        screenshot_path = resolve_screenshot(dataset_root, ad)
        # combine individual labels into broad perception scores
        negative_score = _sum_labels(label_dist, NEGATIVE_LABELS)
        positive_score = _sum_labels(label_dist, POSITIVE_LABELS)
        rows.append(
            {
                "ad_id": int(ad["id"]),
                "cluster_id": ad.get("cluster_id", ""),
                "screenshot": ad.get("screenshot", ""),
                "screenshot_path": str(screenshot_path),
                "screenshot_exists": screenshot_path.exists(),
                "content_labels": ";".join(map(str, content_labels)),
                "risk_domain_hint": _risk_domain(list(map(str, content_labels))),
                "negative_perception_score": negative_score,
                "positive_perception_score": positive_score,
                "clickbait": float(label_dist.get("clickbait", 0.0)),
                "deceptive": float(label_dist.get("deceptive", 0.0)),
                "manipulative": float(label_dist.get("manipulative", 0.0)),
                "distasteful": float(label_dist.get("distasteful", 0.0)),
                "unclear": float(label_dist.get("unclear", 0.0)),
                "mean_rating": float(sum(ratings) / len(ratings)) if ratings else None,
                "num_ratings": len(ratings),
                "num_comments": len(ad.get("comments", []) or []),
            }
        )
    # keep the final inventory in stable id order
    return pd.DataFrame(rows).sort_values("ad_id").reset_index(drop=True)
