from __future__ import annotations

import argparse
from pathlib import Path
from bad_ads_regulation.dataset import build_inventory


def main() -> None:
    # define the dataset folder and output file
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--out", default="outputs/dataset_inventory.csv")
    args = parser.parse_args()

    # scan the dataset and build one inventory row per ad
    inventory = build_inventory(args.dataset_root)

    # save the inventory and show screenshot coverage
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    inventory.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"Wrote {len(inventory)} rows to {out}")
    print(f"Screenshots found: {int(inventory['screenshot_exists'].sum())}/{len(inventory)}")


if __name__ == "__main__":
    main()
