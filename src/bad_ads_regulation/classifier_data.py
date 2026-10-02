from __future__ import annotations

from pathlib import Path
import pandas as pd
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

# define the fixed columns shared by all classifier stages
MANIFEST_COLUMNS = ["ad_id", "screenshot_path", "label", "split", "ocr_text", "ocr_status"]

# load one transformed image text and label for each dataset item
class ClassifierDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, image_transform) -> None:
        self.frame = frame.reset_index(drop=True).copy()
        self.image_transform = image_transform

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> dict[str, object]:
        # open the image only while reading the current item
        row = self.frame.iloc[index]
        with Image.open(row["screenshot_path"]) as image:
            image_tensor = self.image_transform(image.convert("RGB"))
        return {
            "ad_id": str(row["ad_id"]),
            "text": str(row.get("ocr_text", "")),
            "image": image_tensor,
            "label": torch.tensor(float(row["label"]), dtype=torch.float32),
        }


#   create one immutable 60/20/20 advertisement-level split
def stratified_split(frame: pd.DataFrame, seed: int) -> pd.DataFrame:
    # check the input before creating the fixed split
    required = {"ad_id", "screenshot_path", "label"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Missing split columns: {sorted(missing)}")
    if frame["ad_id"].astype(str).duplicated().any():
        raise ValueError("ad_id must be unique before splitting.")
    if frame["label"].nunique() != 2:
        raise ValueError("Binary labels are required for stratified splitting.")

    # reserve 40 percent then divide it evenly between validation and test
    train, remainder = train_test_split(
        frame,
        test_size=0.4,
        stratify=frame["label"],
        random_state=seed,
    )
    validation, test = train_test_split(
        remainder,
        test_size=0.5,
        stratify=remainder["label"],
        random_state=seed,
    )
    return pd.concat(
        [
            train.assign(split="train"),
            validation.assign(split="validation"),
            test.assign(split="test"),
        ],
        ignore_index=True,
    )


# keep only independent training inputs and the weak binary target
def build_training_manifest(labels: pd.DataFrame, seed: int) -> pd.DataFrame:
    # accept either a prepared label or the original annotation field
    label_column = "label" if "label" in labels.columns else "is_potential_violation"
    required = {"ad_id", "screenshot_path", label_column}
    missing = required - set(labels.columns)
    if missing:
        raise ValueError(f"Missing manifest columns: {sorted(missing)}")

    # keep only independent inputs and the binary weak target
    base = labels[["ad_id", "screenshot_path", label_column]].copy()
    base = base.rename(columns={label_column: "label"})
    base["ad_id"] = base["ad_id"].astype(str)
    base["label"] = pd.to_numeric(base["label"], errors="raise").astype(int)
    if not base["label"].isin([0, 1]).all():
        raise ValueError("Weak target labels must be 0 or 1.")

    # create the split and add empty ocr fields for the next stage
    split = stratified_split(base, seed=seed)
    split["ocr_text"] = ""
    split["ocr_status"] = "not_run"
    return split[MANIFEST_COLUMNS].sort_values("ad_id", key=lambda s: s.astype(int)).reset_index(drop=True)


def read_csv_flexible(path: str | Path) -> pd.DataFrame:
    # try common csv encodings used by project outputs
    for encoding in ("utf-8-sig", "utf-8", "gb18030", "gbk", "cp936"):
        try:
            return pd.read_csv(path, encoding=encoding, keep_default_na=False)
        except UnicodeDecodeError:
            continue
    raise RuntimeError(f"Could not decode CSV: {path}")
