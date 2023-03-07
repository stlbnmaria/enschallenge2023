from pathlib import Path
import time

import math
import numpy as np
import pandas as pd


def pred_aggregation(
    values: np.array, agg_over: np.array, agg_by: str = "mean"
) -> pd.DataFrame:
    """
    This function aggregates predicted or true values by some aggregation form (e.g. mean)
    and over some common feature, e.g. samples id or patient id.
    """
    agg_unique = np.unique(agg_over)

    if agg_by == "mean":
        preds = {sample: [np.mean(values[agg_over == sample])] for sample in agg_unique}
    elif agg_by == "median":
        preds = {sample: [np.median(values[agg_over == sample])] for sample in agg_unique}
    elif agg_by == "max":
        preds = {sample: [np.max(values[agg_over == sample])] for sample in agg_unique}
    elif agg_by == "min":
        preds = {sample: [np.min(values[agg_over == sample])] for sample in agg_unique}
    elif agg_by.startswith("mean_"):
        bound = int(agg_by.split("_")[1]) / 100
        preds = {}
        for sample in agg_unique:
            temp = values[agg_over == sample]
            idx = (-temp).argsort()[: math.ceil(len(temp) * bound)]
            preds[sample] = [np.mean(temp[idx])]

    df = pd.DataFrame(preds)
    df = df.transpose().reset_index()
    df.columns = ["Sample ID", "Target"]
    return df


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