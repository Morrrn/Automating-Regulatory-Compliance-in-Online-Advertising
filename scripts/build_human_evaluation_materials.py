from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd

# define the fields used in the blind review stage
BLIND_COLUMNS = [
    "assessor_id",
    "assessment_date",
    "ad_id",
    "screenshot_path",
    "content_labels",
    "negative_perception_score",
    "human_is_potential_concern",
    "human_primary_code_area",
    "human_secondary_code_areas",
    "human_cap_code_rule_refs",
    "human_confidence",
    "human_needs_human_review",
    "human_rationale",
    "uncertainty_or_missing_context",
]

# define the fields used to rate model explanations
EXPLANATION_COLUMNS = [
    "assessor_id",
    "assessment_date",
    "ad_id",
    "screenshot_path",
    "model_is_potential_violation",
    "model_risk_score",
    "model_primary_code_area",
    "model_secondary_code_areas",
    "model_cap_code_rule_refs",
    "model_confidence",
    "model_rationale",
    "explanation_evidence_groundedness_1_to_5",
    "explanation_code_mapping_appropriateness_1_to_5",
    "explanation_clarity_and_actionability_1_to_5",
    "explanation_overall_status",
    "explanation_comments",
]

def build_materials(candidates_csv: str | Path, out_dir: str | Path) -> None:
    # load the candidate ads and create the output folder
    candidates = pd.read_csv(candidates_csv, keep_default_na=False, encoding="utf-8-sig")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # build the blind sheet without showing model outputs
    blind = pd.DataFrame()
    blind["assessor_id"] = ""
    blind["assessment_date"] = ""
    for column in ("ad_id", "screenshot_path", "content_labels", "negative_perception_score"):
        blind[column] = candidates.get(column, "")
    for column in BLIND_COLUMNS[len(blind.columns):]:
        blind[column] = ""
    blind = blind[BLIND_COLUMNS]
    blind.to_csv(out_dir / "human_evaluation_blind_decision_sheet.csv", index=False, encoding="utf-8-sig")

    # build the second sheet with model outputs and empty rating fields
    explanation = pd.DataFrame()
    explanation["assessor_id"] = ""
    explanation["assessment_date"] = ""
    explanation["ad_id"] = candidates["ad_id"]
    explanation["screenshot_path"] = candidates.get("screenshot_path", "")
    explanation["model_is_potential_violation"] = candidates.get("is_potential_violation", "")
    explanation["model_risk_score"] = candidates.get("risk_score", "")
    explanation["model_primary_code_area"] = candidates.get("primary_code_area", "")
    explanation["model_secondary_code_areas"] = candidates.get("secondary_code_areas", "")
    explanation["model_cap_code_rule_refs"] = candidates.get("cap_code_rule_refs", "")
    explanation["model_confidence"] = candidates.get("confidence", "")
    explanation["model_rationale"] = candidates.get("rationale", "")
    for column in EXPLANATION_COLUMNS[len(explanation.columns):]:
        explanation[column] = ""
    explanation = explanation[EXPLANATION_COLUMNS]
    explanation.to_csv(out_dir / "human_evaluation_explanation_quality_sheet.csv", index=False, encoding="utf-8-sig")

    print(f"Created materials for {len(candidates)} advertisements in {out_dir}")


def main() -> None:
    # define the input and output settings
    parser = argparse.ArgumentParser(description="Build blinded human-evaluation sheets from full-result sample candidates.")
    parser.add_argument(
        "--candidates",
        default="outputs/full_results_analysis/human_review_sample_candidates.csv",
    )
    parser.add_argument("--out-dir", default="outputs/human_evaluation")
    args = parser.parse_args()
    build_materials(args.candidates, args.out_dir)


if __name__ == "__main__":
    main()
