from __future__ import annotations

import re
from pathlib import Path
import pandas as pd


def normalise_ocr(value: str) -> str:
    # collapse repeated space in extracted text
    return re.sub(r"\s+", " ", value).strip()


def build_reader(use_gpu: bool):
    # load the optional ocr dependency only when it is needed
    import easyocr

    return easyocr.Reader(["en"], gpu=use_gpu, verbose=False)


def extract_manifest_ocr(manifest: pd.DataFrame, reader) -> pd.DataFrame:
    # check that every row can be linked to an image
    required = {"ad_id", "screenshot_path"}
    missing = required - set(manifest.columns)
    if missing:
        raise ValueError(f"Manifest missing OCR columns: {sorted(missing)}")

    # copy the manifest so the original data stays unchanged
    result = manifest.copy()
    texts: list[str] = []
    statuses: list[str] = []
    errors: list[str] = []
    # extract text and record a status for each image
    for row in result.itertuples(index=False):
        image_path = Path(getattr(row, "screenshot_path"))
        try:
            if not image_path.is_file():
                raise FileNotFoundError(str(image_path))
            chunks = reader.readtext(str(image_path), detail=0, paragraph=False)
            value = normalise_ocr(" ".join(chunks))
            texts.append(value)
            statuses.append("ok" if value else "empty")
            errors.append("")
        except Exception as exc:  # keep ocr failures visible in the output
            texts.append("")
            statuses.append("error")
            errors.append(f"{type(exc).__name__}: {exc}")
    # add text status errors and character counts to the manifest
    result["ocr_text"] = texts
    result["ocr_status"] = statuses
    result["ocr_error"] = errors
    result["ocr_characters"] = result["ocr_text"].str.len()
    return result
