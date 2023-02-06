from pathlib import Path
import time
from tqdm import tqdm

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score


def load_train_data(metadata: pd.DataFrame, data_path=Path("../storage/")):
    """
    This function loads the MoCov features for training.
    """
    train_features_dir = data_path / "train_input" / "moco_features"

    X_train = []
    y_train = []
    centers_train = []
    patients_train = []

    for sample, label, center, patient in tqdm(
        metadata[["Sample ID", "Target", "Center ID", "Patient ID"]].values
    ):
        # load the coordinates and features (1000, 3+2048)
        _features = np.load(train_features_dir / sample)
        # get coordinates (zoom level, tile x-coord on the slide, tile y-coord on the slide)
        # and the MoCo V2 features
        _, features = _features[:, :3], _features[:, 3:]  # Ks
        # slide-level averaging
        X_train.append(np.mean(features, axis=0))
        y_train.append(label)
        centers_train.append(center)
        patients_train.append(patient)

    # convert to numpy arrays
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    patients_train = np.array(patients_train)
    patients_unique = np.unique(patients_train)
    y_unique = np.array(
        [np.mean(y_train[patients_train == p]) for p in patients_unique]
    )
    return X_train, y_train, patients_unique, y_unique, patients_train


def train_mocov_features(
    model, X_train, y_train, patients_unique, y_unique, patients_train
):
    """
    This function trains any model of 5-fold cv on the mocov features 
    and returns a list of models (one for every fold).
    """
    aucs = []
    lrs = []
    # 5-fold CV is repeated 5 times with different random states
    for k in range(5):
        kfold = StratifiedKFold(5, shuffle=True, random_state=k)
        fold = 0
        # split is performed at the patient-level
        for train_idx_, val_idx_ in kfold.split(patients_unique, y_unique):
            # retrieve the indexes of the samples corresponding to the
            # patients in `train_idx_` and `test_idx_`
            train_idx = np.arange(len(X_train))[
                pd.Series(patients_train).isin(patients_unique[train_idx_])
            ]
            val_idx = np.arange(len(X_train))[
                pd.Series(patients_train).isin(patients_unique[val_idx_])
            ]
            # set the training and validation folds
            X_fold_train = X_train[train_idx]
            y_fold_train = y_train[train_idx]
            X_fold_val = X_train[val_idx]
            y_fold_val = y_train[val_idx]
            # instantiate the model
            lr = model
            # fit it
            lr.fit(X_fold_train, y_fold_train)
            # get the predictions (1-d probability)
            preds_val = lr.predict_proba(X_fold_val)[:, 1]
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


def load_avg_test_mocov_features(data_path=Path("../storage/")):
    test_features_dir = data_path / "test_input" / "moco_features"

    md_test = pd.read_csv(data_path / "supplementary_data" / "test_metadata.csv")

    X_test = []

    for sample in tqdm(md_test["Sample ID"].values):
        _features = np.load(test_features_dir / sample)
        _, features = _features[:, :3], _features[:, 3:]
        X_test.append(np.mean(features, axis=0))

    return np.array(X_test)


def predict_cv_classifiers(lrs: list):
    """
    This function takes a list of classifiers trained on crossvalidation,
    predicts the target for every cv-classifier and averages over this
    prediction to create the final prediction.
    """
    X_test = load_avg_test_mocov_features()

    preds_test = 0
    # loop over the classifiers
    for lr in lrs:
        preds_test += lr.predict_proba(X_test)[:, 1]
    # and take the average (ensembling technique)
    preds_test = preds_test / len(lrs)
    return preds_test


def store_submission(
    preds: np.array,
    sub_name: str,
    data_path=Path("../storage"),
    submission_path=Path("../submissions"),
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
        submission_path / f"{timestamp}_{sub_name}_output.csv", index=None
    )
