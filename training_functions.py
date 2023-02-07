from pathlib import Path
from tqdm import tqdm

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score


# TODO: try not to average tiles before modeling, but afterwards for every patient for prediction
# TODO: aggregate classifier differently? - e.g. instead of average use second layer model
# TODO: incorporate the different centers to take into account heterogenity
# TODO: try different ML models and tune them


def load_train_data(metadata: pd.DataFrame, data_path=Path("./storage/")):
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


def load_avg_test_mocov_features(data_path=Path("./storage/")):
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
