import os
from datetime import datetime
from pathlib import Path

import math
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, ParameterGrid, train_test_split
from sklearn.utils._testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import roc_auc_score
from sklearn.base import clone
from sklearn.ensemble import StackingClassifier

from modeling.tabular_models import get_tabular_estimator, read_grid_tuning
from utils import pred_aggregation, load_mocov_train_data


@ignore_warnings(category=ConvergenceWarning)
def train_mocov_features(
    model,
    X_train,
    y_train,
    samples_train,
    centers_train,
    agg_by: str = "mean",
    tile_avg: str = None,
) -> list:
    """
    This function trains any model of 3-fold center cv on the mocov features
    and returns a list of models (one for every fold).
    """
    aucs = []
    lrs = []

    # split is performed at the center level
    kfold = GroupKFold(n_splits=3)
    for train_idx_, val_idx_ in kfold.split(X_train, y_train, centers_train):
        # set the training and validation data
        X_fold_train = X_train[train_idx_]
        y_fold_train = y_train[train_idx_]
        samples_fold = samples_train[train_idx_]
        X_fold_val = X_train[val_idx_]
        y_fold_val = y_train[val_idx_]
        val_center = np.unique(centers_train[val_idx_])[0]

        # instantiate the model
        lr = clone(model)
        lr = lr.fit(X_fold_train, y_fold_train)

        # get the predictions (1-d probability)
        preds_train = lr.predict_proba(X_fold_train)[:, 1]
        preds_val = lr.predict_proba(X_fold_val)[:, 1]

        # compute sample level predictions if MoCo features are not aggregated
        if tile_avg is None:
            preds_train = pred_aggregation(preds_train, samples_fold, agg_by)
            y_fold_train = pred_aggregation(y_fold_train, samples_fold, agg_by)
            train_y = pd.merge(y_fold_train, preds_train, on="Sample ID")
            preds_train = train_y["Target_y"]
            y_fold_train = train_y["Target_x"]

            samples_val = samples_train[val_idx_]
            preds_val = pred_aggregation(preds_val, samples_val, agg_by)
            y_fold_val = pred_aggregation(y_fold_val, samples_val, agg_by)
            val_y = pd.merge(y_fold_val, preds_val, on="Sample ID")
            preds_val = val_y["Target_y"]
            y_fold_val = val_y["Target_x"]

        # compute the AUC score using scikit-learn
        train_auc = roc_auc_score(y_fold_train, preds_train)
        test_auc = roc_auc_score(y_fold_val, preds_val)
        print(
            f"AUC on validation center {val_center}: Train - {train_auc:.3f}, Val - {test_auc:.3f}"
        )
        # add AUC und classifier to list
        aucs.append(test_auc)
        lrs.append(lr)
    print("----------------------------")
    print(
        f"3-fold cross-validated AUC averaged: "
        f"{np.mean(aucs):.3f} ({np.std(aucs):.3f})"
    )
    return lrs


def train_tabular(
    model: str,
    agg_by: str,
    tile_avg: str = None,
    scaling: str = None,
    drop: bool = True,
    feat_select: bool = True,
    n_jobs: int = 6,
    data_path=Path("./storage/"),
) -> list:
    """
    This function initilializes training for the tabular data.
    """
    # get estimator and set params with grid
    grid = read_grid_tuning()
    estimator = get_tabular_estimator(model, n_jobs)
    if grid is not None:
        estimator.set_params(**grid)

    # load the train data
    (X_train, y_train, _, samples_train, centers_train,) = load_mocov_train_data(
        data_path=data_path,
        tile_averaging=tile_avg,
        scaling=scaling,
        drop_dupes=drop,
        feat_select=feat_select,
    )
    # perform training and extract cv classifiers
    lrs = train_mocov_features(
        estimator,
        X_train,
        y_train,
        samples_train,
        centers_train,
        agg_by,
        tile_avg=tile_avg,
    )
    return lrs


@ignore_warnings(category=ConvergenceWarning)
def tuning_moco(
    model: str,
    agg_by: str = "mean",
    tile_avg: str = None,
    scaling: str = None,
    drop: bool = True,
    feat_select: bool = True,
    n_jobs: int = 6,
    data_path=Path("./storage/"),
    file_name: str = "cv_results",
) -> None:
    """
    This function performs grid tuning for an estimator and saves the results in a subfolder of modeling.
    """
    # define the input varibales
    (X_train, y_train, _, samples_train, centers_train,) = load_mocov_train_data(
        data_path=data_path,
        tile_averaging=tile_avg,
        scaling=scaling,
        drop_dupes=drop,
        feat_select=feat_select,
    )
    grid = ParameterGrid(read_grid_tuning())
    out_path = os.path.join("./modeling", model)
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    timestamp = datetime.today().strftime("%Y%m%d_%H%M")
    cv_results = {
        "grid": [],
        "train_AUC_C2_C5": [],
        "train_AUC_C1_C2": [],
        "train_AUC_C1_C5": [],
        "val_AUC_C1": [],
        "val_AUC_C5": [],
        "val_AUC_C2": [],
    }

    print(f"Started tuning for {model} with grid of size: {len(grid)}")
    for hyp in grid:
        # get estimator and set params, append params to results
        estimator = get_tabular_estimator(model, n_jobs)
        estimator.set_params(**hyp)
        cv_results["grid"].append(hyp)

        # cv split is performed at the center level
        kfold = GroupKFold(n_splits=3)
        for train_idx_, val_idx_ in kfold.split(X_train, y_train, centers_train):
            # set the training and validation folds
            X_fold_train = X_train[train_idx_]
            y_fold_train = y_train[train_idx_]
            samples_fold = samples_train[train_idx_]
            X_fold_val = X_train[val_idx_]
            y_fold_val = y_train[val_idx_]
            samples_val = samples_train[val_idx_]
            val_center = np.unique(centers_train[val_idx_])[0].replace("_", "")

            # fit the model
            estimator.fit(X_fold_train, y_fold_train)

            # get the predictions (1-d probability)
            preds_train = estimator.predict_proba(X_fold_train)[:, 1]
            preds_val = estimator.predict_proba(X_fold_val)[:, 1]

            # compute sample level predictions if MoCo features are not aggregated
            if tile_avg is None:
                preds_train = pred_aggregation(preds_train, samples_fold, agg_by)
                y_fold_train = pred_aggregation(y_fold_train, samples_fold, agg_by)
                train_y = pd.merge(y_fold_train, preds_train, on="Sample ID")
                preds_train = train_y["Target_y"]
                y_fold_train = train_y["Target_x"]

                preds_val = pred_aggregation(preds_val, samples_val, agg_by)
                y_fold_val = pred_aggregation(y_fold_val, samples_val, agg_by)
                val_y = pd.merge(y_fold_val, preds_val, on="Sample ID")
                preds_val = val_y["Target_y"]
                y_fold_val = val_y["Target_x"]

            # compute the AUC score using scikit-learn
            train_auc = roc_auc_score(y_fold_train, preds_train)
            test_auc = roc_auc_score(y_fold_val, preds_val)

            # append cv results
            train_centers = {"C1": "C2_C5", "C2": "C1_C5", "C5": "C1_C2"}
            cv_results["train_AUC_" + train_centers[val_center]].append(train_auc)
            cv_results["val_AUC_" + val_center].append(test_auc)

        # saving cv_results at every round of the grid
        results = pd.DataFrame(cv_results)
        results["train_AUC_mean"] = results[
            ["train_AUC_C2_C5", "train_AUC_C1_C2", "train_AUC_C1_C5"]
        ].mean(axis=1)
        results["val_AUC_mean"] = results[
            ["val_AUC_C1", "val_AUC_C2", "val_AUC_C5"]
        ].mean(axis=1)
        results.to_csv(
            os.path.join(out_path, f"{timestamp}_{file_name}.csv"), index=False
        )

        print(f"Done with tuning iteration {list(grid).index(hyp) + 1} / {len(grid)}")

    print(f"----------- GridSearchCV results saved successfully-----------")

    best_val_score = max(results["val_AUC_mean"])
    print(f"----------- Best avg. validation AUC: {best_val_score:.3f} -----------")


@ignore_warnings(category=ConvergenceWarning)
def stacking_estimators(
    models: list,
    tile_avg: str = None,
    scaling: str = None,
    drop: bool = True,
    feat_select: bool = True,
    n_jobs: int = 6,
    data_path=Path("./storage/"),
) -> list:
    """
    This function trains a stacked generalization with 90% of the data used for the base
    models and 10% for the second layer. It returns a list of stacked generalizations for a
    center fold cv split.
    """
    # load the train data
    (X_train, y_train, _, _, centers_train) = load_mocov_train_data(
        data_path=data_path,
        tile_averaging=tile_avg,
        scaling=scaling,
        drop_dupes=drop,
        feat_select=feat_select,
    )
    aucs = []
    lrs = []
    estimators = [(model, get_tabular_estimator(model, n_jobs)) for model in models]

    # split is performed at the center level
    kfold = GroupKFold(n_splits=3)
    for train_idx_, val_idx_ in kfold.split(X_train, y_train, centers_train):
        # get the 90-10 training split for the center fold train data to fit model
        X_base, X_second, y_base, y_second = train_test_split(
            X_train[train_idx_],
            y_train[train_idx_],
            test_size=0.1,
            random_state=0,
            stratify=centers_train[train_idx_],
        )
        # set validation data
        X_fold_val = X_train[val_idx_]
        y_fold_val = y_train[val_idx_]
        val_center = np.unique(centers_train[val_idx_])[0]

        # fit the base models and second layer
        estimators = [
            (model[0], clone(model[1]).fit(X_base, y_base)) for model in estimators
        ]
        stack = StackingClassifier(
            estimators, cv="prefit", stack_method="predict_proba"
        )
        stack.fit(X_second, y_second)

        # get the predictions (1-d probability)
        preds_val = stack.predict_proba(X_fold_val)[:, 1]

        # compute the AUC score using scikit-learn
        test_auc = roc_auc_score(y_fold_val, preds_val)
        print(f"AUC validation center {val_center}: Val - {test_auc:.3f}")
        # add AUC and classifier to lists
        aucs.append(test_auc)
        lrs.append(stack)
    print("----------------------------")
    print(
        f"3-fold cross-validated AUC averaged: "
        f"{np.mean(aucs):.3f} ({np.std(aucs):.3f})"
    )
    return lrs
