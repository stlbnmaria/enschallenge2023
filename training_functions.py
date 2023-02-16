from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, GroupKFold
from sklearn.metrics import roc_auc_score

from modeling.tabular_models import get_tabular_estimator


# TODO: aggregate classifier differently? - e.g. instead of average use second layer model
# TODO: incorporate the different centers to take into account heterogenity
# TODO: try different ML models and tune them


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
    tile_avg: bool = True,
    rep_cv: int = 5,
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
        # split is performed at the patient-level
        for train_idx_, val_idx_ in kfold.split(X_train, y_train, centers_train):
            # retrieve the indexes of the samples corresponding to the
            # patients in `train_idx_` and `test_idx_`
            # train_idx = np.arange(len(X_train))[
            #     pd.Series(patients_train).isin(patients_unique[train_idx_])
            # ]
            # val_idx = np.arange(len(X_train))[
            #     pd.Series(patients_train).isin(patients_unique[val_idx_])
            # ]
            # set the training and validation folds
            X_fold_train = X_train[train_idx_]
            y_fold_train = y_train[train_idx_]
            X_fold_val = X_train[val_idx_]
            y_fold_val = y_train[val_idx_]
            # instantiate the model
            lr = model
            # fit it
            lr.fit(X_fold_train, y_fold_train)

            # get the predictions (1-d probability)
            preds_val = lr.predict_proba(X_fold_val)[:, 1]

            if not tile_avg:
                samples_val = samples_train[val_idx_]
                preds_val = [
                    np.mean(preds_val[samples_val == sample])
                    for sample in np.unique(samples_val)
                ]
                y_fold_val = [
                    np.mean(y_fold_val[samples_val == sample])
                    for sample in np.unique(samples_val)
                ]

            # compute the AUC score using scikit-learn
            auc = roc_auc_score(y_fold_val, preds_val)
            print(f"AUC on split {k} fold {fold}: {auc:.3f}")
            aucs.append(auc)
            # add the logistic regression to the list of classifiers
            lrs.append(lr)
            fold += 1
        print("----------------------------")
    print(
        f"5-fold cross-validated AUC averaged over {k+1} repeats: "
        f"{np.mean(aucs):.3f} ({np.std(aucs):.3f})"
    )
    return lrs


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


def predict_cv_classifiers(lrs: list, tile_avg: bool = True):
    """
    This function takes a list of classifiers trained on crossvalidation,
    predicts the target for every cv-classifier and averages over this
    prediction to create the final prediction.
    """
    X_test, _, _, samples_test = load_mocov_test_data(tile_averaging=tile_avg)
    samples_unique = np.unique(samples_test)

    preds_test = 0
    # loop over the classifiers
    for lr in lrs:
        if tile_avg:
            preds_test += lr.predict_proba(X_test)[:, 1]
        else:
            temp = lr.predict_proba(X_test)[:, 1]
            temp = np.array(
                [np.mean(temp[samples_test == sample]) for sample in samples_unique]
            )
            assert temp.shape[0] == (149)
            preds_test += temp

    # and take the average (ensembling technique)
    preds_test = preds_test / len(lrs)
    return preds_test


def train_tabular(model: str):
    """
    This function trains the tabular data.
    """
    estimator = get_tabular_estimator(model)

    (
        X_train,
        y_train,
        patients_unique,
        y_unique,
        patients_train,
        samples_train,
        centers_train,
    ) = load_mocov_train_data(tile_averaging=False)
    lrs = train_mocov_features(
        estimator,
        X_train,
        y_train,
        patients_unique,
        y_unique,
        patients_train,
        samples_train,
        centers_train,
        tile_avg=False,
    )
    preds = predict_cv_classifiers(lrs, tile_avg=False)
    return preds
