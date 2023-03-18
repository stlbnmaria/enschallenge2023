from pathlib import Path

import math
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler, OneHotEncoder


def load_mocov_train_data(
    data_path=Path("./storage/"), 
    tile_averaging: str = None, 
    scaling: str = None, 
    onehot_zoom: bool = False, 
    drop_dupes: bool = True,
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
    coords = metadata[:, 4].astype(float)

    if drop_dupes:
        df = pd.DataFrame({"patient": patients_train, 
                            "zoom": coords})
        df = df.loc[::1000, :]
        # replace 15 with 17 
        df.loc[df.zoom==15, "zoom"] = 17
        # sort and keep only last (16 zoom levels)
        df = df.sort_values(by='zoom', ascending=False)
        df = df.drop_duplicates(subset='patient', keep="last")
        idx = df.sort_index().index

        idx = np.array([y for i in idx for y in list(range(i, i + 1000))])

        feat = feat[idx]
        y_train = y_train[idx]
        patients_train = patients_train[idx]
        samples_train = samples_train[idx]
        centers_train = centers_train[idx]
        coords = coords[idx]
        assert idx.shape[0] == 305_000


    if onehot_zoom:
        enc = OneHotEncoder(categories=[[14., 15., 16., 17.]])
        coords = enc.fit_transform(coords.reshape(-1, 1))
        coords = coords.toarray()[:, 1:]

    # set default X_train
    X_train = np.column_stack((coords, feat))
    if tile_averaging == "pos_avg":
        X_train = np.where(X_train==0, np.nan, X_train)

    if scaling is not None:
        # scale the feature values for each center seperately
        X_train = np.empty([0, 2048])
        if tile_averaging == "pos_avg":
            # change zeros to nan so that they are ignored during scaling
            feat = np.where(feat==0, np.nan, feat)
        for center in np.unique(centers_train):
            scaler = {"MinMax": MinMaxScaler(), "Standard": StandardScaler()}[scaling]
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
        coords = np.hstack([coords[c1], coords[c2], coords[c5]])
        X_train = np.column_stack((coords, X_train))
        centers_train = np.sort(centers_train)

    if tile_averaging is not None:
        if tile_averaging == "pos_avg":
            # aggregate the MoCo features by taking the mean for every sample
            temp = np.empty([0, 2049])
            for sample in samples_train[::1000]:
                features = np.nanmean(X_train[samples_train == sample], axis=0)
                temp = np.vstack((temp, features))
            X_train = np.where(np.isnan(temp), 0, temp)
        elif tile_averaging == "avg":
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
    data_path=Path("./storage/"), tile_averaging: str = None, scaling: str = None, onehot_zoom: bool = False
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
    coords = metadata[:, 3].astype(float)

    if onehot_zoom:
        enc = OneHotEncoder(categories=[[14., 15., 16., 17.]])
        coords = enc.fit_transform(coords.reshape(-1, 1))
        coords = coords.toarray()[:, 1:]

    # set default X_test
    X_test = np.column_stack((coords, feat))
    if tile_averaging == "pos_avg":
        X_test = np.where(X_test==0, np.nan, X_test)

    if scaling is not None:
        # scale the feature values for each center seperately
        X_test = np.empty([0, 2048])
        if tile_averaging == "pos_avg":
            # change zeros to nan so that they are ignored during scaling
            feat = np.where(feat==0, np.nan, feat)
        for center in np.unique(centers_test):
            scaler = {"MinMax": MinMaxScaler(), "Standard": StandardScaler()}[scaling]
            X_test = np.vstack(
                [X_test, scaler.fit_transform(feat[centers_test == center])]
            )

        # conditions for the three centers to reorder the other arrays
        c3 = centers_test == "C_3"
        c4 = centers_test == "C_4"

        # reorder patients, samples and centers arrays
        patients_test = np.hstack([patients_test[c3], patients_test[c4]])
        samples_test = np.hstack([samples_test[c3], samples_test[c4]])
        coords = np.hstack([coords[c3], coords[c4]])
        X_test = np.column_stack((coords, X_test))
        centers_test = np.sort(centers_test)

    if tile_averaging is not None:
        if tile_averaging == "pos_avg":
            # aggregate the MoCo features by taking the mean for every sample
            temp = np.empty([0, 2049])
            for sample in samples_test[::1000]:
                features = np.nanmean(X_test[samples_test == sample], axis=0)
                temp = np.vstack((temp, features))
            X_test = np.where(np.isnan(temp), 0, temp)
        elif tile_averaging == "avg":
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
