from pathlib import Path
import time
from typing import Union

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.base import clone
from sklearn.ensemble import StackingClassifier

from utils import load_mocov_test_data, pred_aggregation, load_mocov_train_data
from modeling.tabular_models import read_grid_tuning, get_tabular_estimator


def train_for_submission(
    model: Union[str, list],
    agg_by: str,
    n_jobs: int = 6,
    tile_avg: str = None,
    scaling: str = None,
    drop: bool = True,
    feat_select: bool = True,
) -> pd.DataFrame:
    """
    This function trains an estimator or a list of estimators as stacking
    on the whole train data and predicts the probability for the test data set.
    """
    # load train data
    X_train, y_train, _, _, centers_train = load_mocov_train_data(
        tile_averaging=tile_avg,
        scaling=scaling,
        drop_dupes=drop,
        feat_select=feat_select,
    )

    # load test data
    X_test, _, samples_test, _ = load_mocov_test_data(
        tile_averaging=tile_avg,
        scaling=scaling,
        feat_select=feat_select,
    )

    if isinstance(model, list):
        # set the training folds for the base models and second layer estimator
        X_base, X_second, y_base, y_second = train_test_split(
            X_train, y_train, test_size=0.1, random_state=0, stratify=centers_train
        )

        # get estimators from string
        estimators = [(estim, get_tabular_estimator(estim, n_jobs)) for estim in model]

        # perform fit on base models
        estimators = [
            (estim[0], clone(estim[1]).fit(X_base, y_base)) for estim in estimators
        ]

        # fit second layer model
        estimator = StackingClassifier(
            estimators, cv="prefit", stack_method="predict_proba"
        )
        estimator.fit(X_second, y_second)

    else:
        # load estimator and grid
        grid = read_grid_tuning()
        estimator = get_tabular_estimator(model, n_jobs)
        if grid is not None:
            estimator.set_params(**grid)
        # fit model
        estimator.fit(X_train, y_train)

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
) -> None:
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
