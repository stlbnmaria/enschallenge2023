from pathlib import Path

import math
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def load_mocov_train_data(
    data_path=Path("./storage/"), tile_averaging: bool = False, scaling: bool = False
):
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

    # set default X_train
    X_train = feat.copy()

    if scaling:
        # scale the feature values for each center seperately
        X_train = np.empty([0, 2048])
        for center in np.unique(centers_train):
            scaler = StandardScaler()
            X_train = np.vstack(
                [X_train, scaler.fit_transform(feat[centers_train == center])]
            )

        # conditions for the three centers to reorder the other arrays
        c1 = centers_train == "C_1"
        c2 = centers_train == "C_2"
        c5 = centers_train == "C_5"

        # reorder y, patients, samples and centers arrays
        y_train = np.hstack([y_train[c1], y_train[c2], y_train[c5]])
        patients_train = np.hstack(
            [patients_train[c1], patients_train[c2], patients_train[c5]]
        )
        samples_train = np.hstack(
            [samples_train[c1], samples_train[c2], samples_train[c5]]
        )
        centers_train = np.sort(centers_train)

    if tile_averaging:
        # aggregate the MoCo features by taking the mean for every sample
        X_train = [
            np.mean(X_train[samples_train == sample], axis=0)
            for sample in samples_train[::1000]
        ]
        X_train = np.array(X_train)

        # reduce the oversampled arrays
        y_train = y_train[::1000]
        patients_train = patients_train[::1000]
        centers_train = centers_train[::1000]
        samples_train = samples_train[::1000]

    return (
        X_train,
        y_train,
        patients_train,
        samples_train,
        centers_train,
    )


def load_mocov_test_data(
    data_path=Path("./storage/"), tile_averaging: bool = False, scaling: bool = False
):
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

    # set default X_test
    X_test = feat.copy()

    if scaling:
        # scale the feature values for each center seperately
        X_test = np.empty([0, 2048])
        for center in np.unique(centers_test):
            scaler = StandardScaler()
            X_test = np.vstack(
                [X_test, scaler.fit_transform(feat[centers_test == center])]
            )

        # conditions for the three centers to reorder the other arrays
        c3 = centers_test == "C_3"
        c4 = centers_test == "C_4"

        # reorder patients, samples and centers arrays
        patients_test = np.hstack(
            [patients_test[c3], patients_test[c4]]
        )
        samples_test = np.hstack(
            [samples_test[c3], samples_test[c4]]
        )
        centers_test = np.sort(centers_test)

    if tile_averaging:
        # aggregate the MoCo features by taking the mean for every sample
        X_test = [
            np.mean(X_test[samples_test == sample], axis=0)
            for sample in samples_test[::1000]
        ]
        X_test = np.array(X_test)

        # reduce the oversampled arrays
        patients_test = patients_test[::1000]
        centers_test = centers_test[::1000]
        samples_test = samples_test[::1000]

    return X_test, patients_test, samples_test, centers_test


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
        preds = {
            sample: [np.median(values[agg_over == sample])] for sample in agg_unique
        }
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
