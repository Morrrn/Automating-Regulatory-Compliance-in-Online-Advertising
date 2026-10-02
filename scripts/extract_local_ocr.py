from __future__ import annotations

import argparse
from pathlib import Path
import torch

from bad_ads_regulation.classifier_data import read_csv_flexible
from bad_ads_regulation.classifier_ocr import build_reader, extract_manifest_ocr


def main() -> None:
    # define input output and ocr runtime settings
    parser = argparse.ArgumentParser(description="Extract local EasyOCR text for classifier training without VLM leakage.")
    parser.add_argument("--manifest", default="outputs/multimodal_baselines/classifier_manifest.csv")
    parser.add_argument("--out", default="outputs/multimodal_baselines/classifier_manifest_with_ocr.csv")
    parser.add_argument("--limit", type=int, default=None, help="Process only the first N rows for a smoke test.")
    parser.add_argument("--cpu", action="store_true", help="Disable GPU even when CUDA is available.")
    args = parser.parse_args()

    # load the manifest and optionally keep a smaller test sample
    manifest = read_csv_flexible(args.manifest)
    if args.limit is not None:
        manifest = manifest.head(args.limit).copy()
    # use the gpu when available unless cpu mode is requested
    use_gpu = torch.cuda.is_available() and not args.cpu
    reader = build_reader(use_gpu=use_gpu)

    # extract text from each image in the manifest
    result = extract_manifest_ocr(manifest, reader)

    # save the updated manifest
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(out, index=False, encoding="utf-8-sig")
    # print(f"EasyOCR GPU enabled: {use_gpu}")
    print(f"Wrote {len(result)} rows to {out}")


if __name__ == "__main__":
    main()
