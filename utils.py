from pathlib import Path

import math
import numpy as np
import pandas as pd


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
