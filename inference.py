from pathlib import Path
import time

import numpy as np
import pandas as pd

from utils import load_mocov_test_data, pred_aggregation

def predict_cv_classifiers(
    lrs: list, agg_by: str, tile_avg: bool = True, data_path=Path("./storage/")
):
    """
    This function takes a list of classifiers trained on crossvalidation,
    predicts the target for every cv-classifier and averages over this
    prediction to create the final prediction.
    """
    X_test, _, _, samples_test = load_mocov_test_data(
        data_path=data_path, tile_averaging=tile_avg
    )

    preds_test = 0
    # loop over the classifiers
    for lr in lrs:
        if tile_avg:
            preds_test += lr.predict_proba(X_test)[:, 1]
        else:
            temp = lr.predict_proba(X_test)[:, 1]
            temp = pred_aggregation(temp, samples_test, agg_by)["Target"]
            assert temp.shape[0] == (149)
            preds_test += temp

    # and take the average (ensembling technique)
    preds_test = preds_test / len(lrs)
    return preds_test


def store_submission(
    preds: np.array,
    sub_name: str,
    data_path=Path("./storage"),
    submission_path=Path("./submissions"),
):
    """
    This functions combines the sample ids in the test data set and the
    predictions from an ML model to save a csv that can be directly uploaded
    to the submission platform.
    """
    md_test = pd.read_csv(data_path / "supplementary_data" / "test_metadata.csv")

    submission = pd.DataFrame(
        {"Sample ID": md_test["Sample ID"].values, "Target": preds}
    ).sort_values(
        "Sample ID"
    )  # extra step to sort the sample IDs

    # sanity checks
    assert all(submission["Target"].between(0, 1)), "`Target` values must be in [0, 1]"
    assert submission.shape == (
        149,
        2,
    ), "Your submission file must be of shape (149, 2)"
    assert list(submission.columns) == [
        "Sample ID",
        "Target",
    ], "Your submission file must have columns `Sample ID` and `Target`"

    # save the submission as a csv file
    t = time.localtime()
    timestamp = time.strftime("%Y-%m-%d-%H%M", t)
    submission.to_csv(
        submission_path / f"{timestamp}_{sub_name}.csv", index=None
    )