import os
from datetime import datetime
from pathlib import Path

import math
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, GroupKFold, ParameterGrid
from sklearn.utils._testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import roc_auc_score

from modeling.tabular_models import get_tabular_estimator, read_grid_tuning
from utils import pred_aggregation


def load_mocov_train_data(data_path=Path("./storage/"), tile_averaging: bool = True):
    """
    This function loads the MoCov features full file for training and
    performs averaging over the tiles per sample as default.
    """
    input = np.load(data_path / "train_input" / "mocov_features_train.npz")
    metadata = input["metadata"]
    feat = input["features"].astype(float)
    y_train = metadata[:, 0].astype(float)
    patients_train = metadata[:, 1]
    samples_train = metadata[:, 2]
    centers_train = metadata[:, 3]

    if tile_averaging:
        X_train = [
            np.mean(feat[samples_train == sample], axis=0)
            for sample in np.unique(samples_train)
        ]
        X_train = np.array(X_train)

        y_train = [
            np.unique(y_train[samples_train == sample])
            for sample in np.unique(samples_train)
        ]
        y_train = np.array(y_train).flatten()

        patients_train = [
            np.unique(patients_train[samples_train == sample])
            for sample in np.unique(samples_train)
        ]
        patients_train = np.array(patients_train).flatten()

        centers_train = [
            np.unique(centers_train[samples_train == sample])
            for sample in np.unique(samples_train)
        ]
        centers_train = np.array(centers_train)

        samples_train = np.unique(samples_train)
    else:
        X_train = feat

    patients_unique = np.unique(patients_train)
    y_unique = np.array(
        [np.mean(y_train[patients_train == p]) for p in patients_unique]
    )

    return (
        X_train,
        y_train,
        patients_unique,
        y_unique,
        patients_train,
        samples_train,
        centers_train,
    )


def train_mocov_features(
    model,
    X_train,
    y_train,
    patients_unique,
    y_unique,
    patients_train,
    samples_train,
    centers_train,
    agg_by: str = "mean",
    tile_avg: bool = True,
    rep_cv: int = 1,
    subsampling: bool = False,
):
    """
    This function trains any model of 5-fold cv on the mocov features
    and returns a list of models (one for every fold).
    """
    aucs = []
    lrs = []
    # 5-fold CV is repeated 5 times with different random states
    for k in range(rep_cv):
        kfold = GroupKFold(n_splits=3)
        fold = 0
        # split is performed at the center level
        for train_idx_, val_idx_ in kfold.split(X_train, y_train, centers_train):
            # set the training and validation folds
            X_fold_train = X_train[train_idx_]
            y_fold_train = y_train[train_idx_]
            X_fold_val = X_train[val_idx_]
            y_fold_val = y_train[val_idx_]
            samples_fold = samples_train[train_idx_]
            val_center = np.unique(centers_train[val_idx_])[0]

            if subsampling:
                df = pd.DataFrame({"id": samples_fold})
                df.reset_index(inplace=True, drop=True)
                sub_index = df.groupby("id").sample(frac=0.8, random_state=0).index
                X_fold_train = X_fold_train[sub_index]
                y_fold_train = y_train[sub_index]
                samples_fold = samples_fold[sub_index]

            # instantiate the model
            lr = model
            # fit it
            lr.fit(X_fold_train, y_fold_train)

            # get the predictions (1-d probability)
            preds_train = lr.predict_proba(X_fold_train)[:, 1]
            preds_val = lr.predict_proba(X_fold_val)[:, 1]

            if not tile_avg:
                preds_train = pred_aggregation(preds_train, samples_fold, agg_by)
                y_fold_train = pred_aggregation(y_fold_train, samples_fold, agg_by)
                train_y = pd.merge(y_fold_train, preds_train, on='Sample ID')
                preds_train = train_y["Target_y"]
                y_fold_train = train_y["Target_x"]

                samples_val = samples_train[val_idx_]
                preds_val = pred_aggregation(preds_val, samples_val, agg_by)
                y_fold_val = pred_aggregation(y_fold_val, samples_val, agg_by)
                val_y = pd.merge(y_fold_val, preds_val, on='Sample ID')
                preds_val = val_y["Target_y"]
                y_fold_val = val_y["Target_x"]

            # compute the AUC score using scikit-learn
            train_auc = roc_auc_score(y_fold_train, preds_train)
            test_auc = roc_auc_score(y_fold_val, preds_val)
            print(
                f"AUC on split {k} validation center {val_center}: Train - {train_auc:.3f}, Val - {test_auc:.3f}"
            )
            aucs.append(test_auc)
            # add the logistic regression to the list of classifiers
            lrs.append(lr)
            fold += 1
        print("----------------------------")
    print(
        f"5-fold cross-validated AUC averaged over {k+1} repeats: "
        f"{np.mean(aucs):.3f} ({np.std(aucs):.3f})"
    )
    return lrs


@ignore_warnings(category=ConvergenceWarning)
def tuning_moco(
    model: str, agg_by: str = "mean", n_jobs: int = 6, data_path=Path("./storage/")
):
    """
    This function performs grid tuning for an estimator and saves the results in a subfolder of modeling.
    """
    # define the input varibales
    (
        X_train,
        y_train,
        patients_unique,
        y_unique,
        patients_train,
        samples_train,
        centers_train,
    ) = load_mocov_train_data(data_path=data_path, tile_averaging=False)
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
        estimator = get_tabular_estimator(model, n_jobs)
        estimator.set_params(**hyp)
        cv_results["grid"].append(hyp)
        kfold = GroupKFold(n_splits=3)
        # split is performed at the center level
        for train_idx_, val_idx_ in kfold.split(X_train, y_train, centers_train):
            # set the training and validation folds
            X_fold_train = X_train[train_idx_]
            y_fold_train = y_train[train_idx_]
            X_fold_val = X_train[val_idx_]
            y_fold_val = y_train[val_idx_]
            samples_fold = samples_train[train_idx_]
            samples_val = samples_train[val_idx_]
            val_center = np.unique(centers_train[val_idx_])[0].replace("_", "")

            # fit the model
            estimator.fit(X_fold_train, y_fold_train)

            # get the predictions (1-d probability)
            preds_train = estimator.predict_proba(X_fold_train)[:, 1]
            preds_train = pred_aggregation(preds_train, samples_fold, agg_by)
            y_fold_train = pred_aggregation(y_fold_train, samples_fold, agg_by)
            train_y = pd.merge(y_fold_train, preds_train, on='Sample ID')
            preds_train = train_y["Target_y"]
            y_fold_train = train_y["Target_x"]

            preds_val = estimator.predict_proba(X_fold_val)[:, 1]
            preds_val = pred_aggregation(preds_val, samples_val, agg_by)
            y_fold_val = pred_aggregation(y_fold_val, samples_val, agg_by)
            val_y = pd.merge(y_fold_val, preds_val, on='Sample ID')
            preds_val = val_y["Target_y"]
            y_fold_val = val_y["Target_x"]

            # compute the AUC score using scikit-learn
            train_auc = roc_auc_score(y_fold_train, preds_train)
            test_auc = roc_auc_score(y_fold_val, preds_val)

            train_centers = {"C1": "C2_C5", "C2": "C1_C5", "C5": "C1_C2"}
            cv_results["train_AUC_" + train_centers[val_center]].append(train_auc)
            cv_results["val_AUC_" + val_center].append(test_auc)
        print(f"Done with tuning iteration {list(grid).index(hyp) + 1} / {len(grid)}")

    # saving cv_results
    results = pd.DataFrame(cv_results)
    results["train_AUC_mean"] = results[
        ["train_AUC_C2_C5", "train_AUC_C1_C2", "train_AUC_C1_C5"]
    ].mean(axis=1)
    results["val_AUC_mean"] = results[["val_AUC_C1", "val_AUC_C2", "val_AUC_C5"]].mean(
        axis=1
    )
    results.to_csv(os.path.join(out_path, f"{timestamp}_cv_results.csv"), index=False)
    print(f"----------- GridSearchCV results saved successfully-----------")

    best_val_score = max(results["val_AUC_mean"])
    print(f"----------- Best avg. validation AUC: {best_val_score:.3f} -----------")


def load_mocov_test_data(data_path=Path("./storage/"), tile_averaging: bool = True):
    """
    This function loads the MoCov features full file for testing and
    performs averaging over the tiles per sample as default.
    """
    input = np.load(data_path / "test_input" / "mocov_features_test.npz")
    metadata = input["metadata"]
    feat = input["features"].astype(float)
    patients_test = metadata[:, 0]
    samples_test = metadata[:, 1]
    centers_test = metadata[:, 2]

    if tile_averaging:
        X_test = [
            np.mean(feat[samples_test == sample], axis=0)
            for sample in np.unique(samples_test)
        ]
        X_test = np.array(X_test)

        patients_test = [
            np.unique(patients_test[samples_test == sample])
            for sample in np.unique(samples_test)
        ]
        patients_test = np.array(patients_test).flatten()

        centers_test = [
            np.unique(centers_test[samples_test == sample])
            for sample in np.unique(samples_test)
        ]
        centers_test = np.array(centers_test)

        samples_test = np.unique(samples_test)
    else:
        X_test = feat

    patients_unique = np.unique(patients_test)

    return X_test, patients_unique, patients_test, samples_test


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


def train_tabular(
    model: str, agg_by: str, n_jobs: int = 6, data_path=Path("./storage/")
):
    """
    This function trains the tabular data.
    """
    estimator = get_tabular_estimator(model, n_jobs)

    (
        X_train,
        y_train,
        patients_unique,
        y_unique,
        patients_train,
        samples_train,
        centers_train,
    ) = load_mocov_train_data(data_path=data_path, tile_averaging=False)
    lrs = train_mocov_features(
        estimator,
        X_train,
        y_train,
        patients_unique,
        y_unique,
        patients_train,
        samples_train,
        centers_train,
        agg_by,
        tile_avg=False,
        subsampling=False,
    )
    return lrs
