from __future__ import annotations
import argparse
from pathlib import Path
from bad_ads_regulation.classifier_data import build_training_manifest, read_csv_flexible


def main() -> None:
    # define input output and split settings
    parser = argparse.ArgumentParser(description="Build a leakage-safe weakly supervised classifier manifest.")
    parser.add_argument("--labels", default="outputs/openai_batch_annotations_all.csv")
    parser.add_argument("--out", default="outputs/multimodal_baselines/classifier_manifest.csv")
    parser.add_argument("--seed", type=int, default=20260622)
    args = parser.parse_args()

    # build the training manifest from the model labels
    labels = read_csv_flexible(args.labels)
    manifest = build_training_manifest(labels, seed=args.seed)

    # create the output folder and save the manifest
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(out, index=False, encoding="utf-8-sig")
    # show row and class counts for each split
    print(f"Wrote {len(manifest)} rows to {out}")
    print(manifest.groupby("split")["label"].agg(["count", "sum"]).to_string())


if __name__ == "__main__":
    main()
