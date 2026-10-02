from __future__ import annotations

import argparse
import json
from pathlib import Path
import pandas as pd

from bad_ads_regulation.annotation import validate_annotation
from bad_ads_regulation.openai_api import extract_output_text
from bad_ads_regulation.openai_batch import download_file_content, retrieve_batch


def parse_batch_output(path: str | Path, parsed_out: str | Path, report_out: str | Path) -> None:
    # parse each batch response and track its validation result
    parsed_rows = []
    report_rows = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            custom_id = item.get("custom_id", "")
            ad_id = custom_id.replace("ad-", "")
            error = item.get("error")
            response = item.get("response") or {}
            body = response.get("body") or {}
            # keep failed requests in the report and skip annotation parsing
            if error:
                report_rows.append({"ad_id": ad_id, "custom_id": custom_id, 
                                    "status": "batch_error", "parse_ok": False, "schema_ok": False, "errors": json.dumps(error)})
                continue
            # extract and validate the model annotation
            output_text = extract_output_text(body)
            validation = validate_annotation(output_text)
            report_rows.append({
                "ad_id": ad_id,
                "custom_id": custom_id,
                "status": "ok" if validation.schema_ok else "needs_review",
                "parse_ok": validation.parse_ok,
                "schema_ok": validation.schema_ok,
                "errors": "; ".join(validation.errors),
                "status_code": response.get("status_code"),
            })
            # keep any annotation that was parsed successfully
            if validation.parsed is not None:
                parsed_rows.append({"ad_id": ad_id, "custom_id": custom_id, **validation.parsed, "usage": body.get("usage", {})})

    # save parsed annotations and a row-level validation report
    parsed_out = Path(parsed_out)
    parsed_out.parent.mkdir(parents=True, exist_ok=True)
    with parsed_out.open("w", encoding="utf-8") as f:
        for row in parsed_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    pd.DataFrame(report_rows).to_csv(report_out, index=False, encoding="utf-8-sig")
    print(f"Parsed annotations: {len(parsed_rows)}")
    print(f"Wrote parsed JSONL to {parsed_out}")
    print(f"Wrote validation report to {report_out}")


def main() -> None:
    # define download parsing and output settings
    parser = argparse.ArgumentParser(description="Download and parse OpenAI Batch API results.")
    parser.add_argument("--batch-id", default=None)
    parser.add_argument("--job-json", default="outputs/openai_batch_job.json")
    parser.add_argument("--output-jsonl", default="outputs/openai_batch_output.jsonl")
    parser.add_argument("--parsed-out", default="outputs/openai_batch_annotations_parsed.jsonl")
    parser.add_argument("--report-out", default="outputs/openai_batch_validation_report.csv")
    parser.add_argument("--parse-only", action="store_true", help="Parse an existing --output-jsonl without downloading.")
    args = parser.parse_args()

    # download completed results unless an existing file is requested
    output_path = Path(args.output_jsonl)
    if not args.parse_only:
        # load the batch id and confirm the job is complete
        batch_id = args.batch_id
        if not batch_id:
            job = json.loads(Path(args.job_json).read_text(encoding="utf-8"))
            batch_id = job["batch"]["id"]
        batch = retrieve_batch(batch_id)
        if batch.get("status") != "completed":
            raise RuntimeError(f"Batch is not completed yet: status={batch.get('status')}")
        output_file_id = batch.get("output_file_id")
        if not output_file_id:
            raise RuntimeError("Batch completed but output_file_id is missing.")
        download_file_content(output_file_id, output_path)
        print(f"Downloaded output file {output_file_id} to {output_path}")

    # parse the downloaded or existing output file
    parse_batch_output(output_path, args.parsed_out, args.report_out)


if __name__ == "__main__":
    main()
