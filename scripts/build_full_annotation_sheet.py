from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd


def build_full_annotation_sheet(inventory_csv: str | Path, out: str | Path, prompt_version: str = "v2.3") -> pd.DataFrame:
    # copy the inventory and add fields used during annotation
    inventory = pd.read_csv(inventory_csv)
    sheet = inventory.copy()
    sheet.insert(0, "annotation_batch", "full_dataset")
    sheet.insert(1, "prompt_version", prompt_version)
    sheet["ocr_text"] = ""

    # place the main annotation fields before any extra inventory fields
    preferred = [
        "annotation_batch",
        "prompt_version",
        "ad_id",
        "screenshot_path",
        "content_labels",
        "risk_domain_hint",
        "negative_perception_score",
        "positive_perception_score",
        "clickbait",
        "deceptive",
        "manipulative",
        "distasteful",
        "unclear",
        "mean_rating",
        "num_ratings",
        "num_comments",
        "ocr_text",
    ]
    remaining = [c for c in sheet.columns if c not in preferred]
    sheet = sheet[[c for c in preferred if c in sheet.columns] + remaining]

    # create the output folder and save the sheet
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    sheet.to_csv(out, index=False, encoding="utf-8-sig")
    return sheet


def main() -> None:
    # define the input output and prompt version settings
    parser = argparse.ArgumentParser()
    parser.add_argument("--inventory", default="outputs/dataset_inventory.csv")
    parser.add_argument("--out", default="outputs/full_annotation_sheet.csv")
    parser.add_argument("--prompt-version", default="v2.3")
    args = parser.parse_args()
    sheet = build_full_annotation_sheet(args.inventory, args.out, args.prompt_version)
    print(f"Wrote {len(sheet)} rows to {args.out}")
    print(f"Prompt version: {args.prompt_version}")


if __name__ == "__main__":
    main()
