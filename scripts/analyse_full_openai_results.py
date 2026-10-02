from __future__ import annotations

import argparse
import math
from pathlib import Path

import pandas as pd


# try common csv encodings before using latin1
def read_csv_flexible(path: str | Path) -> pd.DataFrame:
    for encoding in ("utf-8-sig", "utf-8", "gbk", "cp936"):
        try:
            return pd.read_csv(path, keep_default_na=False, encoding=encoding)
        except UnicodeDecodeError:
            continue
    return pd.read_csv(path, keep_default_na=False, encoding="latin1")


# split pipe-separated values into one value per row
def explode_pipe(df: pd.DataFrame, col: str) -> pd.DataFrame:
    tmp = df[["ad_id", col]].copy()
    tmp[col] = tmp[col].fillna("").astype(str).str.split("|")
    tmp = tmp.explode(col)
    tmp[col] = tmp[col].fillna("").astype(str).str.strip()
    return tmp[tmp[col] != ""]


# split semicolon-separated labels into one label per row
def explode_semicolon(df: pd.DataFrame, col: str) -> pd.DataFrame:
    tmp = df[["ad_id", col]].copy()
    tmp[col] = tmp[col].fillna("").astype(str).str.split(";")
    tmp = tmp.explode(col)
    tmp[col] = tmp[col].fillna("").astype(str).str.strip()
    return tmp[tmp[col] != ""]


def pct(n: float, d: float) -> float:
    return 0.0 if d == 0 else 100.0 * n / d


def value_counts_table(df: pd.DataFrame, col: str, total: int, name: str | None = None) -> pd.DataFrame:
    out = df[col].value_counts(dropna=False).rename_axis(name or col).reset_index(name="count")
    out["percent"] = out["count"].map(lambda x: pct(x, total))
    return out

# Build a human-review sample from high risk, borderline, low-confidence, and low-risk cases.
def choose_review_sample(df: pd.DataFrame, n: int) -> pd.DataFrame:
    selected: list[pd.DataFrame] = []
    used: set[str] = set()

    # add new cases without selecting the same ad twice
    def add(label: str, frame: pd.DataFrame, k: int) -> None:
        nonlocal selected, used
        if k <= 0 or frame.empty:
            return
        pool = frame[~frame["ad_id"].astype(str).isin(used)].copy()
        if pool.empty:
            return
        take = pool.head(k).copy()
        take["sample_reason"] = label
        used.update(take["ad_id"].astype(str))
        selected.append(take)

    # build groups for different review needs
    high_risk = df.sort_values(["risk_score", "negative_perception_score"], ascending=[False, False])
    borderline = df[(df["risk_score"] >= 0.40) & (df["risk_score"] <= 0.74)].copy()
    borderline["borderline_distance"] = (borderline["risk_score"] - 0.5).abs()
    borderline = borderline.sort_values(["borderline_distance", "negative_perception_score"])
    low_conf = df[df["confidence"].isin(["low", "medium"])].sort_values(["confidence", "risk_score"], ascending=[True, False])
    perception_gap = df[((df["negative_perception_score"] >= 2.0) & (df["is_potential_violation"] == 0)) | ((df["negative_perception_score"] < 1.0) & (df["is_potential_violation"] == 1))].sort_values("negative_perception_score", ascending=False)
    low_risk_controls = df[df["is_potential_violation"] == 0].sort_values(["risk_score", "negative_perception_score"], ascending=[True, True])

    # give each review group part of the sample
    add("high_risk_priority", high_risk, math.ceil(n * 0.25))
    add("borderline_decision", borderline, math.ceil(n * 0.25))
    add("low_or_medium_confidence", low_conf, math.ceil(n * 0.20))
    add("perception_model_gap", perception_gap, math.ceil(n * 0.20))
    current = sum(len(x) for x in selected)
    add("low_risk_control", low_risk_controls, max(0, n - current))

    # fill any empty sample slots with unused ads
    sample = pd.concat(selected, ignore_index=True) if selected else pd.DataFrame()
    if len(sample) < n:
        add("diversity_fill", df.sort_values("ad_id"), n - len(sample))
        sample = pd.concat(selected, ignore_index=True) if selected else pd.DataFrame()
    return sample.head(n)


def analyse(results_csv: str | Path, source_csv: str | Path, out_dir: str | Path, review_sample_size: int) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # load both files and make ad ids use the same type
    results = read_csv_flexible(results_csv)
    source = read_csv_flexible(source_csv)
    results["ad_id"] = results["ad_id"].astype(str)
    source["ad_id"] = source["ad_id"].astype(str)

    # keep only source fields needed for later summaries
    source_cols = [
        "ad_id", "content_labels", "risk_domain_hint", "negative_perception_score",
        "positive_perception_score", "clickbait", "deceptive", "manipulative",
        "distasteful", "unclear", "mean_rating", "num_ratings", "num_comments", "cluster_id",
    ]
    merged = results.merge(source[[c for c in source_cols if c in source.columns]], on="ad_id", how="left", suffixes=("", "_source"))

    # convert analysis fields to numbers and leave invalid values empty
    numeric_cols = [
        "is_potential_violation", "risk_score", "negative_perception_score", "positive_perception_score",
        "clickbait", "deceptive", "manipulative", "distasteful", "unclear", "mean_rating",
        "num_ratings", "num_comments", "usage_input_tokens", "usage_output_tokens", "usage_total_tokens",
    ]
    for col in numeric_cols:
        if col in merged.columns:
            merged[col] = pd.to_numeric(merged[col], errors="coerce")

    total = len(merged)
    merged.to_csv(out_dir / "openai_results_enriched.csv", index=False, encoding="utf-8-sig")

    # summarize decisions, confidence, risk scores, and token use
    overall = pd.DataFrame([
        {"metric": "n_ads", "value": total},
        {"metric": "potential_violations", "value": int((merged["is_potential_violation"] == 1).sum())},
        {"metric": "non_violations", "value": int((merged["is_potential_violation"] == 0).sum())},
        {"metric": "potential_violation_rate_percent", "value": pct((merged["is_potential_violation"] == 1).sum(), total)},
        {"metric": "needs_human_review", "value": int((merged["needs_human_review"].astype(str).str.lower() == "true").sum())},
        {"metric": "mean_risk_score", "value": float(merged["risk_score"].mean())},
        {"metric": "median_risk_score", "value": float(merged["risk_score"].median())},
        {"metric": "low_confidence", "value": int((merged["confidence"] == "low").sum())},
        {"metric": "medium_confidence", "value": int((merged["confidence"] == "medium").sum())},
        {"metric": "high_confidence", "value": int((merged["confidence"] == "high").sum())},
        {"metric": "total_input_tokens", "value": int(merged["usage_input_tokens"].fillna(0).sum())},
        {"metric": "total_output_tokens", "value": int(merged["usage_output_tokens"].fillna(0).sum())},
        {"metric": "total_tokens", "value": int(merged["usage_total_tokens"].fillna(0).sum())},
    ])
    overall.to_csv(out_dir / "summary_overall.csv", index=False, encoding="utf-8-sig")

    # summarize results by cap code area
    primary = value_counts_table(merged, "primary_code_area", total)
    primary.to_csv(out_dir / "summary_primary_code_area.csv", index=False, encoding="utf-8-sig")

    confidence_by_decision = pd.crosstab(merged["confidence"], merged["is_potential_violation"], margins=True)
    confidence_by_decision.to_csv(out_dir / "confidence_by_decision.csv", encoding="utf-8-sig")

    area_stats = merged.groupby("primary_code_area", dropna=False).agg(
        ads=("ad_id", "count"),
        potential_violations=("is_potential_violation", "sum"),
        mean_risk=("risk_score", "mean"),
        median_risk=("risk_score", "median"),
        mean_negative_perception=("negative_perception_score", "mean"),
        mean_clickbait=("clickbait", "mean"),
        mean_deceptive=("deceptive", "mean"),
        mean_manipulative=("manipulative", "mean"),
    ).reset_index()
    area_stats["violation_rate_percent"] = area_stats.apply(lambda r: pct(r["potential_violations"], r["ads"]), axis=1)
    area_stats = area_stats.sort_values(["potential_violations", "mean_risk"], ascending=[False, False])
    area_stats.to_csv(out_dir / "summary_by_primary_code_area.csv", index=False, encoding="utf-8-sig")

    # split multi-value fields before counting evidence and rule references
    evidence = explode_pipe(merged, "evidence_type").merge(merged[["ad_id", "risk_score", "is_potential_violation"]], on="ad_id", how="left")
    evidence_stats = evidence.groupby("evidence_type").agg(
        ads=("ad_id", "count"),
        mean_risk=("risk_score", "mean"),
        potential_violations=("is_potential_violation", "sum"),
    ).reset_index().sort_values(["ads", "mean_risk"], ascending=[False, False])
    evidence_stats.to_csv(out_dir / "summary_by_evidence_type.csv", index=False, encoding="utf-8-sig")

    refs = explode_pipe(merged, "cap_code_rule_refs").merge(merged[["ad_id", "risk_score", "is_potential_violation"]], on="ad_id", how="left")
    refs_stats = refs.groupby("cap_code_rule_refs").agg(
        ads=("ad_id", "count"),
        mean_risk=("risk_score", "mean"),
        potential_violations=("is_potential_violation", "sum"),
    ).reset_index().sort_values(["ads", "mean_risk"], ascending=[False, False])
    refs_stats.to_csv(out_dir / "summary_by_cap_rule_ref.csv", index=False, encoding="utf-8-sig")

    # compare results across content labels and clusters
    labels = explode_semicolon(merged, "content_labels").merge(merged[["ad_id", "risk_score", "is_potential_violation", "negative_perception_score"]], 
                                                               on="ad_id", how="left")
    label_stats = labels.groupby("content_labels").agg(
        ads=("ad_id", "count"),
        potential_violations=("is_potential_violation", "sum"),
        mean_risk=("risk_score", "mean"),
        mean_negative_perception=("negative_perception_score", "mean"),
    ).reset_index()
    label_stats["violation_rate_percent"] = label_stats.apply(lambda r: pct(r["potential_violations"], r["ads"]), axis=1)
    label_stats = label_stats.sort_values(["ads", "violation_rate_percent", "mean_risk"], ascending=[False, False, False])
    label_stats.to_csv(out_dir / "summary_by_content_label.csv", index=False, encoding="utf-8-sig")

    cluster_stats = merged.groupby("cluster_id", dropna=False).agg(
        ads=("ad_id", "count"),
        potential_violations=("is_potential_violation", "sum"),
        mean_risk=("risk_score", "mean"),
        mean_negative_perception=("negative_perception_score", "mean"),
    ).reset_index()
    cluster_stats["violation_rate_percent"] = cluster_stats.apply(lambda r: pct(r["potential_violations"], r["ads"]), axis=1)
    cluster_stats = cluster_stats.sort_values(["violation_rate_percent", "ads"], ascending=[False, False])
    cluster_stats.to_csv(out_dir / "summary_by_cluster.csv", index=False, encoding="utf-8-sig")

    # check how perception scores relate to model results
    perception_cols = ["negative_perception_score", "positive_perception_score", 
                       "clickbait", "deceptive", "manipulative", "distasteful", "unclear", "mean_rating"]
    corr_rows = []
    for col in perception_cols:
        if col in merged.columns:
            subset = merged[["risk_score", "is_potential_violation", col]].dropna()
            if len(subset) >= 3:
                corr_rows.append({
                    "perception_metric": col,
                    "pearson_with_risk_score": subset["risk_score"].corr(subset[col], method="pearson"),
                    "spearman_with_risk_score": subset["risk_score"].rank().corr(subset[col].rank(), method="pearson"),
                    "pearson_with_violation_decision": subset["is_potential_violation"].corr(subset[col], method="pearson"),
                    "n": len(subset),
                })
    correlations = pd.DataFrame(corr_rows)
    correlations.to_csv(out_dir / "perception_correlation_summary.csv", index=False, encoding="utf-8-sig")

    # save high-risk ads and a mixed sample for human review
    high_risk = merged.sort_values("risk_score", ascending=False).head(30)
    high_risk.to_csv(out_dir / "top_30_high_risk_ads.csv", index=False, encoding="utf-8-sig")

    review_cols = [
        "sample_reason", "ad_id", "screenshot_path", "risk_score", "is_potential_violation", "primary_code_area",
        "secondary_code_areas", "confidence", "negative_perception_score", "content_labels", "cap_code_rule_refs", "rationale",
    ]
    review_sample = choose_review_sample(merged, review_sample_size)
    review_sample[[c for c in review_cols if c in review_sample.columns]].to_csv(out_dir / "human_review_sample_candidates.csv", 
                                                                                 index=False, encoding="utf-8-sig")

    print(f"Wrote analysis outputs to {out_dir}")
    print(f"Rows analysed: {total}")
    print(f"Potential violations: {int((merged['is_potential_violation'] == 1).sum())}")
    print(f"Human review sample: {len(review_sample)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyse full OpenAI Batch annotation results for dissertation reporting.")
    parser.add_argument("--results-csv", default="outputs/openai_batch_annotations_all.csv")
    parser.add_argument("--source-csv", default="outputs/full_annotation_sheet.csv")
    parser.add_argument("--out-dir", default="outputs/full_results_analysis")
    parser.add_argument("--review-sample-size", type=int, default=50)
    args = parser.parse_args()
    analyse(args.results_csv, args.source_csv, args.out_dir, args.review_sample_size)


if __name__ == "__main__":
    main()
