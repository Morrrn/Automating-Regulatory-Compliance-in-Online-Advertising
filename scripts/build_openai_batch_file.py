from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import pandas as pd
from bad_ads_regulation.openai_api import build_responses_payload, preview_payload

# find the default prompt in the project support files
DEFAULT_PROMPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "annotation_prompt_and_schema"
    / "final_openai_annotation_prompt_v2_3.md"
)

# try common csv encodings before using latin1
def read_sheet(path: str | Path) -> pd.DataFrame:
    for encoding in ("utf-8-sig", "utf-8", "gbk", "cp936"):
        try:
            return pd.read_csv(path, keep_default_na=False, encoding=encoding)
        except UnicodeDecodeError:
            continue
    return pd.read_csv(path, keep_default_na=False, encoding="latin1")


def build_batch_file(
    *,
    sheet_path: str | Path,
    prompt_path: str | Path,
    out_jsonl: str | Path,
    preview_jsonl: str | Path | None,
    model: str,
    image_detail: str,
    offset: int = 0,
    limit: int | None = None,
) -> tuple[int, int]:
    # select the requested rows from the annotation sheet
    sheet = read_sheet(sheet_path)
    if offset:
        sheet = sheet.iloc[offset:]
    if limit is not None:
        sheet = sheet.head(limit)
    # load the prompt and prepare the output files
    prompt = Path(prompt_path).read_text(encoding="utf-8-sig").lstrip("\ufeff")
    out_jsonl = Path(out_jsonl)
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    preview_file = Path(preview_jsonl) if preview_jsonl else None
    if preview_file:
        preview_file.parent.mkdir(parents=True, exist_ok=True)

    # write one batch request for each ad
    count = 0
    with out_jsonl.open("w", encoding="utf-8") as out_f:
        prev_f = preview_file.open("w", encoding="utf-8") if preview_file else None
        try:
            for row in sheet.to_dict(orient="records"):
                ad_id = row["ad_id"]
                image_path = row.get("screenshot_path") or row.get("image_path")
                payload = build_responses_payload(
                    prompt=prompt,
                    ad_id=ad_id,
                    image_path=image_path,
                    model=model,
                    ocr_text=str(row.get("ocr_text", "")),
                    image_detail=image_detail,
                )
                line = {
                    "custom_id": f"ad-{ad_id}",
                    "method": "POST",
                    "url": "/v1/responses",
                    "body": payload,
                }
                out_f.write(json.dumps(line, ensure_ascii=False) + "\n")
                # save a readable request copy when preview output is enabled
                if prev_f:
                    preview = {"custom_id": line["custom_id"], "method": line["method"], "url": line["url"], "body": preview_payload(payload)}
                    prev_f.write(json.dumps(preview, ensure_ascii=False) + "\n")
                count += 1
        # close the optional preview file if request building stops early
        finally:
            if prev_f:
                prev_f.close()
    return count, out_jsonl.stat().st_size


def main() -> None:
    # define settings for building batch requests without uploading them
    parser = argparse.ArgumentParser(description="Build an OpenAI Batch API JSONL file for full ad annotation.")
    parser.add_argument("--sheet", default="outputs/full_annotation_sheet.csv")
    parser.add_argument(
        "--prompt",
        default=os.environ.get("CAP_ANNOTATION_PROMPT_PATH", str(DEFAULT_PROMPT_PATH)),
        help=(
            "Path to the annotation prompt. Defaults to the submitted v2.3 prompt; "
            "override with --prompt or CAP_ANNOTATION_PROMPT_PATH."
        ),
    )
    parser.add_argument("--out", default="outputs/openai_batch_input_full.jsonl")
    parser.add_argument("--preview-out", default="outputs/openai_batch_input_full_preview.jsonl")
    parser.add_argument("--model", default="gpt-5.4-mini")
    parser.add_argument("--image-detail", choices=["low", "high", "auto"], default="low")
    parser.add_argument("--offset", type=int, default=0, help="Number of rows to skip before applying --limit.")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    count, size = build_batch_file(
        sheet_path=args.sheet,
        prompt_path=args.prompt,
        out_jsonl=args.out,
        preview_jsonl=args.preview_out,
        model=args.model,
        image_detail=args.image_detail,
        offset=args.offset,
        limit=args.limit,
    )
    print(f"Wrote {count} requests to {args.out}")
    print(f"File size: {size / (1024 * 1024):.2f} MB")
    print(f"Preview written to {args.preview_out}")
    print(f"Prompt: {Path(args.prompt).expanduser().resolve()}")


if __name__ == "__main__":
    main()



