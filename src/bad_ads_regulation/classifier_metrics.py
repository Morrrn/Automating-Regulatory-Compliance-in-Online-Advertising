from __future__ import annotations

import math
from collections.abc import Sequence
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support, roc_auc_score


# calculate one consistent metric set for weak-label binary evaluation
def binary_metrics(
    targets: Sequence[int],
    predictions: Sequence[int],
    probabilities: Sequence[float] | None = None,
) -> dict[str, float | int]:    
    # check that all metric inputs have the same length
    if len(targets) != len(predictions):
        raise ValueError("targets and predictions must have equal length.")
    if probabilities is not None and len(targets) != len(probabilities):
        raise ValueError("targets and probabilities must have equal length.")

    # calculate confusion counts and threshold-based metrics
    tn, fp, fn, tp = confusion_matrix(targets, predictions, labels=[0, 1]).ravel()
    precision, recall, f1, _ = precision_recall_fscore_support(
        targets,
        predictions,
        average="binary",
        zero_division=0,
    )
    # calculate auroc only when scores and both target classes are present
    auroc = math.nan
    if probabilities is not None and len(set(targets)) == 2:
        auroc = float(roc_auc_score(targets, probabilities))
    return {
        "n": len(targets),
        "accuracy": float(accuracy_score(targets, predictions)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "auroc": auroc,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def prediction_table(
    ad_ids: Sequence[str], targets: Sequence[int], probabilities: Sequence[float]
) -> pd.DataFrame:
    # check that every output row has an id target and probability
    if not (len(ad_ids) == len(targets) == len(probabilities)):
        raise ValueError("ad_ids, targets, and probabilities must have equal length.")
    # apply the standard threshold and build the prediction table
    return pd.DataFrame(
        {
            "ad_id": [str(ad_id) for ad_id in ad_ids],
            "weak_label": [int(target) for target in targets],
            "probability": [float(probability) for probability in probabilities],
            "prediction": [int(probability >= 0.5) for probability in probabilities],
        }
    )
