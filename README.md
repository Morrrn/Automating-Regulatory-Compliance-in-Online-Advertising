# Policy-aligned triage of online advertisements

This repository archives code and selected research artifacts for an MSc project on using the UK CAP Code to prioritise online advertisements for human review. The system produces *potential concern* signals, not legal findings of non-compliance.

## What is included

| Path | Contents |
| --- | --- |
| `src/bad_ads_regulation/` | Dataset, annotation, API, classifier, and explanation modules |
| `scripts/` | Pipeline entry points for inventory, annotation, analysis, training, evaluation, and explanations |
| `annotation_prompt_and_schema/` | Final v2.3 prompt and output schema |
| `processed_experiment_data/` | 500-ad annotations, analysis inputs, 50-case human evaluation, fixed classifier split, selected cases, and test metrics |
| `human_evaluation_template/` | Blank human-review forms |

The dissertation PDF, original 500 advertising screenshots, participant-level source data, generated model weights, API response archives, and development-only tests are **not** redistributed here. The source dataset is available from the [University of Washington Ad Perceptions Dataset](https://badads.cs.washington.edu/ad-perceptions-dataset/table.html). Its original publication is E. Zeng, T. Kohno, and F. Roesner, "What Makes a 'Bad' Ad? User Perceptions of Problematic Online Advertising," *CHI 2021*, doi: [10.1145/3411764.3445459](https://doi.org/10.1145/3411764.3445459). Check the dataset's terms before downloading or redistributing it.

The published CSVs use relative screenshot paths of the form `data/chi-bad-ads-data-main/screenshots/<ad_id>.webp`. The screenshot files must be placed there before image-based annotation, OCR, training, or explanation generation. Run commands from the repository root so these paths resolve correctly. No original image is needed to inspect the structured CSV results.

## Environment

Python 3.10 or newer is required. The multimodal requirements were recorded for the original experiment; installing the pinned PyTorch stack may require a platform-specific wheel and substantial disk space.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e .
python -m pip install -r requirements-multimodal.txt
$env:PYTHONPATH = (Resolve-Path .\src).Path
```

For API annotation, set `OPENAI_API_KEY` in your shell or environment before a live submission. Never put a key in a tracked file. The included code defaults to `gpt-5.4-mini` and the archived final prompt; model availability and API pricing may differ at the time of rerun.

## Reproducing the main stages

1. Download the source dataset separately and place its `screenshots/` directory under `data/chi-bad-ads-data-main/`. For a fresh inventory, retain the dataset's other source files in that same directory.
2. Inspect the recorded split in `processed_experiment_data/classifier_manifest_with_ocr.csv`. Its OCR text has been removed from this public copy. The fixed 60/20/20 split has 300 training, 100 validation, and 100 held-out test advertisements.
3. Build a small API request preview, if desired:

```powershell
python scripts/build_openai_batch_file.py --sheet processed_experiment_data/full_annotation_sheet.csv --out outputs/batch_preview.jsonl --preview-out outputs/batch_preview_readable.jsonl --limit 5
python scripts/submit_openai_batch.py --input outputs/batch_preview.jsonl --dry-run
```

Submitting a real Batch API job incurs charges. The archived annotation results in `processed_experiment_data/openai_batch_annotations_all.csv` can be used without repeating that step.

4. Recreate OCR text locally from the separately downloaded screenshots, then train each baseline on the recorded weak labels. Model downloads and training may take time; checkpoints are generated under `outputs/runs/`.

```powershell
python scripts/extract_local_ocr.py --manifest processed_experiment_data/classifier_manifest_with_ocr.csv --out outputs/classifier_manifest_with_ocr.csv
python scripts/train_multimodal_baselines.py --model text --manifest outputs/classifier_manifest_with_ocr.csv --output-dir outputs/runs
python scripts/train_multimodal_baselines.py --model image --manifest outputs/classifier_manifest_with_ocr.csv --output-dir outputs/runs
python scripts/train_multimodal_baselines.py --model fusion --manifest outputs/classifier_manifest_with_ocr.csv --output-dir outputs/runs
python scripts/evaluate_multimodal_baselines.py --manifest outputs/classifier_manifest_with_ocr.csv --run-dir outputs/runs --output-dir outputs/evaluation
```

The recorded held-out metrics are in `processed_experiment_data/test_metrics_comparison.csv`: text F1 0.735, image F1 0.574, and late-fusion F1 0.699. They evaluate agreement with VLM weak labels, not verified CAP Code breaches. Structured human-review ratings remain in the included data.

## Notes on reproducibility and scope

- The final prompt and schema are the versions used for the full annotation run; API outputs may vary on rerun.
- CSV screenshot paths were rebased from machine-specific absolute paths to repository-relative paths. Free-text OCR, ad summaries, rationales, reviewer comments, and assessor metadata were removed from the public data. Structured labels, splits, ratings, and metrics were retained.
- No API credential value or local machine path is included. The environment-variable name in the code is only a placeholder for a credential supplied privately at runtime.
- The raw advertising material and participant responses were intentionally excluded from this public archive. Follow the source dataset citation and any applicable terms for access.
- The project is research software for human-in-the-loop triage. It does not establish whether an advertiser holds substantiation or whether an ad is legally non-compliant.
