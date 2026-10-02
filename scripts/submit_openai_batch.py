from __future__ import annotations

import argparse
import json
from pathlib import Path
from bad_ads_regulation.openai_batch import create_batch, upload_batch_file


def main() -> None:
    # define the input metadata run mode and output file
    parser = argparse.ArgumentParser(description="Upload a Batch API JSONL input file and create a batch job.")
    parser.add_argument("--input", default="outputs/openai_batch_input_full.jsonl")
    parser.add_argument("--metadata-description", default="bad-ads CAP Code annotation full dataset")
    parser.add_argument("--dry-run", action="store_true", help="Only print what would be uploaded/submitted.")
    parser.add_argument("--out", default="outputs/openai_batch_job.json")
    args = parser.parse_args()

    # show the planned request without uploading in dry-run mode
    input_path = Path(args.input)
    if args.dry_run:
        print(f"DRY RUN: would upload {input_path} ({input_path.stat().st_size / (1024 * 1024):.2f} MB) with purpose=batch")
        print("DRY RUN: would create batch endpoint=/v1/responses completion_window=24h")
        return

    # upload the request file and create the batch job
    file_obj = upload_batch_file(input_path)
    batch = create_batch(
        file_obj["id"],
        endpoint="/v1/responses",
        metadata={"description": args.metadata_description},
    )
    # save the returned file and batch details
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"file": file_obj, "batch": batch}, indent=2), encoding="utf-8")
    print(f"Uploaded file: {file_obj['id']}")
    print(f"Created batch: {batch['id']} status={batch.get('status')}")
    print(f"Saved metadata to {out}")


if __name__ == "__main__":
    main()
