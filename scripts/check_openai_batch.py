from __future__ import annotations

import argparse
import json
from pathlib import Path
from bad_ads_regulation.openai_batch import retrieve_batch

def main() -> None:
    # define where to read job details and save the current status
    parser = argparse.ArgumentParser(description="Check an OpenAI Batch API job status.")
    parser.add_argument("--batch-id", default=None)
    parser.add_argument("--job-json", default="outputs/openai_batch_job.json")
    parser.add_argument("--out", default="outputs/openai_batch_status.json")
    args = parser.parse_args()

    # use the given batch id or load it from the saved job details
    batch_id = args.batch_id
    if not batch_id:
        job = json.loads(Path(args.job_json).read_text(encoding="utf-8"))
        batch_id = job["batch"]["id"]

    # fetch the latest batch status
    batch = retrieve_batch(batch_id)

    # save the full status and print its key fields
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(batch, indent=2), encoding="utf-8")
    print(f"Batch: {batch['id']}")
    print(f"Status: {batch.get('status')}")
    print(f"Request counts: {batch.get('request_counts')}")
    print(f"Output file: {batch.get('output_file_id')}")
    print(f"Error file: {batch.get('error_file_id')}")
    print(f"Saved status to {out}")


if __name__ == "__main__":
    main()
