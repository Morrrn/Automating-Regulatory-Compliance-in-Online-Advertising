from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

# try common encodings used by manually edited csv files
def read_csv_flexible(path: str | Path) -> tuple[pd.DataFrame, str]:
    for encoding in ("utf-8-sig", "utf-8", "gb18030", "gbk", "cp936"):
        try:
            return pd.read_csv(path, encoding=encoding, keep_default_na=False), encoding
        except UnicodeDecodeError:
            continue
    raise RuntimeError(f"Could not decode CSV: {path}")

# remove index columns created when csv files are saved
def clean_columns(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.loc[:, ~frame.columns.str.match(r"^Unnamed")].copy()

# measure agreement after allowing for agreement by chance
def cohen_kappa(a: pd.Series, b: pd.Series) -> float:
    observed = float((a == b).mean())
    expected = sum(float((a == label).mean()) * float((b == label).mean()) for label in sorted(set(a) | set(b)))
    return (observed - expected) / (1 - expected) if expected < 1 else 1.0


def analyze(blind_path: str | Path, explanation_path: str | Path, out_dir: str | Path) -> None:
    # load both review sheets and remove unused index columns
    blind, blind_encoding = read_csv_flexible(blind_path)
    explanation, explanation_encoding = read_csv_flexible(explanation_path)
    blind = clean_columns(blind)
    explanation = clean_columns(explanation)

    # make ad ids use the same type in both sheets
    for frame in (blind, explanation):
        frame["ad_id"] = frame["ad_id"].astype(str)

    # require one matching row per ad in both review sheets
    if blind["ad_id"].duplicated().any() or explanation["ad_id"].duplicated().any():
        raise ValueError("Duplicate ad_id values prevent one-to-one evaluation analysis.")
    if set(blind["ad_id"]) != set(explanation["ad_id"]):
        raise ValueError("Blind and explanation sheets have different advertisement ID sets.")

    # join the blind decisions with the explanation ratings
    merged = blind.merge(explanation, on=["ad_id", "screenshot_path"], how="inner", suffixes=("_blind", "_explanation"))
    merged["human_decision"] = pd.to_numeric(merged["human_is_potential_concern"], errors="raise").astype(int)
    merged["model_decision"] = pd.to_numeric(merged["model_is_potential_violation"], errors="raise").astype(int)
    merged["decision_agreement"] = merged["human_decision"] == merged["model_decision"]
    merged["primary_area_agreement"] = merged["human_primary_code_area"] == merged["model_primary_code_area"]

    # calculate binary agreement and classification metrics
    n = len(merged)
    tp = int(((merged["human_decision"] == 1) & (merged["model_decision"] == 1)).sum())
    tn = int(((merged["human_decision"] == 0) & (merged["model_decision"] == 0)).sum())
    fp = int(((merged["human_decision"] == 0) & (merged["model_decision"] == 1)).sum())
    fn = int(((merged["human_decision"] == 1) & (merged["model_decision"] == 0)).sum())
    # report kappa because raw agreement can be affected by class balance
    agreement = float(merged["decision_agreement"].mean())
    kappa = cohen_kappa(merged["human_decision"], merged["model_decision"])
    precision = tp / (tp + fp) if tp + fp else float("nan")
    recall = tp / (tp + fn) if tp + fn else float("nan")
    specificity = tn / (tn + fp) if tn + fp else float("nan")
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else float("nan")

    # build confusion tables for decisions and cap code areas
    decision_confusion = pd.crosstab(
        merged["human_decision"], merged["model_decision"], margins=True,
        rownames=["human_decision"], colnames=["model_decision"],
    )
    primary_confusion = pd.crosstab(
        merged["human_primary_code_area"], merged["model_primary_code_area"],
        rownames=["human_primary_code_area"], colnames=["model_primary_code_area"],
    )
    primary_exact = float(merged["primary_area_agreement"].mean())
    both_positive = merged[(merged["human_decision"] == 1) & (merged["model_decision"] == 1)]
    primary_exact_both_positive = float(both_positive["primary_area_agreement"].mean()) if not both_positive.empty else float("nan")

    # summarize the three explanation quality scores
    score_columns = [
        "explanation_evidence_groundedness_1_to_5",
        "explanation_code_mapping_appropriateness_1_to_5",
        "explanation_clarity_and_actionability_1_to_5",
    ]
    for col in score_columns:
        merged[col] = pd.to_numeric(merged[col], errors="raise")
    score_summary = pd.DataFrame([
        {
            "dimension": col.replace("explanation_", "").replace("_1_to_5", ""),
            "mean": merged[col].mean(),
            "median": merged[col].median(),
            "min": merged[col].min(),
            "max": merged[col].max(),
        }
        for col in score_columns
    ])
    status_summary = merged["explanation_overall_status"].value_counts().rename_axis("status").reset_index(name="count")
    status_summary["percent"] = 100 * status_summary["count"] / n
    needs_amendment_rows = merged[merged["explanation_overall_status"] == "needs_amendment"]

    # collect decision errors and label them from the human view
    error_cases = merged[~merged["decision_agreement"]].copy()
    error_cases["error_type_relative_to_human"] = error_cases.apply(
        lambda row: "false_positive" if row["model_decision"] == 1 else "false_negative", axis=1
    )
    error_cols = [
        "ad_id", "screenshot_path", "error_type_relative_to_human", "human_decision", "model_decision",
        "human_primary_code_area", "model_primary_code_area", "human_confidence", "model_confidence",
        "human_rationale", "model_rationale", "explanation_overall_status", "explanation_comments",
    ]

    # collect ads where the primary cap code area differs
    area_disagreements = merged[~merged["primary_area_agreement"]].copy()
    area_cols = [
        "ad_id", "screenshot_path", "human_decision", "model_decision", "human_primary_code_area",
        "model_primary_code_area", "human_rationale", "model_rationale",
    ]

    # record checks that affect the reliability of the evaluation
    data_quality = pd.DataFrame([
        {"check": "blind_rows", "value": len(blind)},
        {"check": "explanation_rows", "value": len(explanation)},
        {"check": "id_sets_match", "value": True},
        {"check": "blind_encoding", "value": blind_encoding},
        {"check": "explanation_encoding", "value": explanation_encoding},
        {"check": "blank_assessor_id_blind", "value": int(blind["assessor_id"].astype(str).str.strip().eq("").sum())},
        {"check": "blank_assessment_date_blind", "value": int(blind["assessment_date"].astype(str).str.strip().eq("").sum())},
        {"check": "blank_human_rationale", "value": int(blind["human_rationale"].astype(str).str.strip().eq("").sum())},
        {"check": "disagreement_cases_with_blank_human_rationale", "value": int(error_cases["human_rationale"].astype(str).str.strip().eq("").sum())},
        {"check": "needs_amendment_cases", "value": len(needs_amendment_rows)},
        {"check": "needs_amendment_cases_with_blank_comments", "value": int(needs_amendment_rows["explanation_comments"].astype(str).str.strip().eq("").sum())},
    ])

    # save detailed results and quality checks as separate csv files
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out_dir / "human_evaluation_merged.csv", index=False, encoding="utf-8-sig")
    decision_confusion.to_csv(out_dir / "decision_confusion_matrix.csv", encoding="utf-8-sig")
    primary_confusion.to_csv(out_dir / "primary_area_confusion_matrix.csv", encoding="utf-8-sig")
    score_summary.to_csv(out_dir / "explanation_score_summary.csv", index=False, encoding="utf-8-sig")
    status_summary.to_csv(out_dir / "explanation_status_summary.csv", index=False, encoding="utf-8-sig")
    error_cases[error_cols].to_csv(out_dir / "decision_error_cases.csv", index=False, encoding="utf-8-sig")
    area_disagreements[area_cols].to_csv(out_dir / "primary_area_disagreement_cases.csv", index=False, encoding="utf-8-sig")
    data_quality.to_csv(out_dir / "human_evaluation_data_quality.csv", index=False, encoding="utf-8-sig")

    # save the main agreement metrics in one table
    metrics = pd.DataFrame([
        {"metric": "n", "value": n},
        {"metric": "human_positive_rate", "value": float(merged["human_decision"].mean())},
        {"metric": "model_positive_rate", "value": float(merged["model_decision"].mean())},
        {"metric": "raw_decision_agreement", "value": agreement},
        {"metric": "cohens_kappa", "value": kappa},
        {"metric": "precision_relative_to_human", "value": precision},
        {"metric": "recall_relative_to_human", "value": recall},
        {"metric": "specificity_relative_to_human", "value": specificity},
        {"metric": "f1_relative_to_human", "value": f1},
        {"metric": "primary_area_exact_agreement", "value": primary_exact},
        {"metric": "primary_area_exact_agreement_both_positive", "value": primary_exact_both_positive},
        {"metric": "model_false_positives_relative_to_human", "value": fp},
        {"metric": "model_false_negatives_relative_to_human", "value": fn},
    ])
    metrics.to_csv(out_dir / "human_evaluation_metrics.csv", index=False, encoding="utf-8-sig")

    print(f"Wrote analysis outputs to {out_dir}")
    print(f"Raw agreement={agreement:.3f}; kappa={kappa:.3f}; errors={len(error_cases)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyse blinded human evaluation of automated CAP Code triage.")
    parser.add_argument("--blind", default="outputs/human_evaluation/human_evaluation_blind_decision_sheet.csv")
    parser.add_argument("--explanation", default="outputs/human_evaluation/human_evaluation_explanation_quality_sheet.csv")
    parser.add_argument("--out-dir", default="outputs/human_evaluation_analysis")
    args = parser.parse_args()
    analyze(args.blind, args.explanation, args.out_dir)


if __name__ == "__main__":
    main()
