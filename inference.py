from pathlib import Path
import time

import numpy as np
import pandas as pd

from utils import load_mocov_test_data, pred_aggregation, load_mocov_train_data
from modeling.tabular_models import read_grid_tuning, get_tabular_estimator


def train_for_submission(
    model: str,
    agg_by: str,
    n_jobs: int = 6,
    tile_avg: str = None,
    scaling: str = None,
    onehot_zoom: bool = False,
    data_path=Path("./storage/"),
):
    """
    This function trains an estimator on the whole train data and
    predicts the probability for the test data set.
    """
    # load estimator and grid
    grid = read_grid_tuning()
    estimator = get_tabular_estimator(model, n_jobs)
    if grid is not None:
        estimator.set_params(**grid)

    # fit the model on the train data
    (
        X_train,
        y_train,
        _,
        _,
        _,
    ) = load_mocov_train_data(tile_averaging=tile_avg, scaling=scaling, onehot_zoom=onehot_zoom)
    estimator.fit(X_train, y_train)

    # load test data
    X_test, _, samples_test, _ = load_mocov_test_data(
        data_path=data_path, tile_averaging=tile_avg, scaling=scaling, onehot_zoom=onehot_zoom
    )

    # preform predictions for averaged or non averaged MoCo features
    if tile_avg is not None:
        preds_test = estimator.predict_proba(X_test)[:, 1]
        preds_test = pd.DataFrame({"Sample ID": samples_test, "Target": preds_test})
    else:
        temp = estimator.predict_proba(X_test)[:, 1]
        preds_test = pred_aggregation(temp, samples_test, agg_by)

    return preds_test


def store_submission(
    preds: pd.DataFrame,
    sub_name: str,
    submission_path=Path("./submissions"),
):
    """
    This functions combines the sample ids in the test data set and the
    predictions from an ML model to save a csv that can be directly uploaded
    to the submission platform.
    """
    submission = preds.sort_values("Sample ID")  # extra step to sort the sample IDs

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
    submission.to_csv(submission_path / f"{timestamp}_{sub_name}.csv", index=None)
