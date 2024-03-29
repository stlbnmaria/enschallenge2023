from pathlib import Path

import math
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler


def load_mocov_train_data(
    data_path=Path("./storage/"),
    tile_averaging: str = None,
    scaling: str = None,
    drop_dupes: bool = True,
    feat_select: bool = True,
) -> tuple[np.array]:
    """
    This function loads the MoCov features full file for training and
    returns X, y, patients, samples and centers.
    Per default, duplicates are droped and feature selection is performed.
    """
    # load numpy input file and feature selection csv
    input = np.load(data_path / "train_input" / "mocov_features_train.npz")
    metadata = input["metadata"]
    feat = input["features"].astype(float)
    y_train = metadata[:, 0].astype(float)
    patients_train = metadata[:, 1]
    samples_train = metadata[:, 2]
    centers_train = metadata[:, 3]
    coords = metadata[:, 4].astype(float)
    imp_idx = np.array(
        pd.read_csv("./modeling/feature_importance/feat_imp_01.csv").iloc[:, 0]
    )

    # drop manually selected duplicates
    if drop_dupes:
        # stack data for duplicate selection
        df = pd.DataFrame({"zoom": coords, "samples": samples_train})
        df = df.loc[::1000, :]
        # specify manual prod list of samples
        drop_list = [
            "ID_014.npy",
            "ID_028.npy",
            "ID_052.npy",
            "ID_117.npy",
            "ID_130.npy",
            "ID_132.npy",
            "ID_233.npy",
            "ID_474.npy",
            "ID_260.npy",
            "ID_296.npy",
            "ID_324.npy",
            "ID_355.npy",
            "ID_440.npy",
            "ID_455.npy",
            "ID_074.npy",
            "ID_083.npy",
            "ID_102.npy",
            "ID_094.npy",
            "ID_243.npy",
            "ID_055.npy",
            "ID_257.npy",
            "ID_272.npy",
            "ID_274.npy",
            "ID_278.npy",
            "ID_279.npy",
            "ID_316.npy",
            "ID_328.npy",
            "ID_334.npy",
            "ID_099.npy",
            "ID_358.npy",
            "ID_371.npy",
            "ID_411.npy",
            "ID_437.npy",
            "ID_460.npy",
            "ID_461.npy",
            "ID_465.npy",
            "ID_481.npy",
            "ID_484.npy",
            "ID_491.npy",
        ]
        # extract index for samples to keep and replicate the original index list for tiles
        df = df[~df.samples.isin(drop_list)]
        idx = df.sort_index().index
        idx = np.array([y for i in idx for y in list(range(i, i + 1000))])

        # filter data according to index - leaving 305 samples
        feat = feat[idx]
        y_train = y_train[idx]
        patients_train = patients_train[idx]
        samples_train = samples_train[idx]
        centers_train = centers_train[idx]
        coords = coords[idx]
        assert idx.shape[0] == 305_000

    # set default X_train
    if tile_averaging == "pos_avg":
        feat = np.where(feat == 0, np.nan, feat)
    X_train = np.column_stack((coords, feat))
    # drop features with low feature importance
    if feat_select:
        X_train = X_train[:, imp_idx]
        feat = feat[:, (imp_idx - 1)]

    # scale the feature values for each center seperately
    if scaling is not None:
        X_train = np.empty([0, feat.shape[1]])
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
        if (0 in imp_idx) or not feat_select:
            coords = np.hstack([coords[c1], coords[c2], coords[c5]])
            X_train = np.column_stack((coords, X_train))
        centers_train = np.sort(centers_train)

    if tile_averaging is not None:
        # aggregate the MoCo features by taking the mean for every sample
        X_train = [
                np.nanmean(X_train[samples_train == sample], axis=0)
                for sample in samples_train[::1000]
            ]
        X_train = np.array(X_train)
        X_train = np.where(np.isnan(X_train), 0, X_train)

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
    data_path=Path("./storage/"),
    tile_averaging: str = None,
    scaling: str = None,
    feat_select: bool = True,
) -> tuple[np.array]:
    """
    This function loads the MoCov features full file for testing and
    returns X, y, patients, and samples. Per default, feature selection is performed.
    """
    # load numpy input file and feature selection csv
    input = np.load(data_path / "test_input" / "mocov_features_test.npz")
    metadata = input["metadata"]
    feat = input["features"].astype(float)
    patients_test = metadata[:, 0]
    samples_test = metadata[:, 1]
    centers_test = metadata[:, 2]
    coords = metadata[:, 3].astype(float)
    imp_idx = np.array(
        pd.read_csv("./modeling/feature_importance/feat_imp_01.csv").iloc[:, 0]
    )

    # set default X_test
    if tile_averaging == "pos_avg":
        feat = np.where(feat == 0, np.nan, feat)
    X_test = np.column_stack((coords, feat))
    # drop features with low feature importance
    if feat_select:
        X_test = X_test[:, imp_idx]
        feat = feat[:, (imp_idx - 1)]

    # scale the feature values for each center seperately
    if scaling is not None:
        X_test = np.empty([0, feat.shape[1]])
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
        if (0 in imp_idx) or not feat_select:
            coords = np.hstack([coords[c3], coords[c4]])
            X_test = np.column_stack((coords, X_test))
        centers_test = np.sort(centers_test)

    if tile_averaging is not None:
        # aggregate the MoCo features by taking the mean for every sample
        X_test = [
                np.nanmean(X_test[samples_test == sample], axis=0)
                for sample in samples_test[::1000]
            ]
        X_test = np.array(X_test)
        X_test = np.where(np.isnan(X_test), 0, X_test)

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
    # get unique values
    agg_unique = np.unique(agg_over)

    # take the mean of predictions
    if agg_by == "mean":
        preds = {sample: [np.mean(values[agg_over == sample])] for sample in agg_unique}
    # take the median of predictions
    elif agg_by == "median":
        preds = {
            sample: [np.median(values[agg_over == sample])] for sample in agg_unique
        }
    # take the maximum of predictions
    elif agg_by == "max":
        preds = {sample: [np.max(values[agg_over == sample])] for sample in agg_unique}
    # take the minimum of predictions
    elif agg_by == "min":
        preds = {sample: [np.min(values[agg_over == sample])] for sample in agg_unique}
    # take an upper percentile mean of predictions
    elif agg_by.startswith("mean_"):
        bound = int(agg_by.split("_")[1]) / 100
        preds = {}
        for sample in agg_unique:
            temp = values[agg_over == sample]
            idx = (-temp).argsort()[: math.ceil(len(temp) * bound)]
            preds[sample] = [np.mean(temp[idx])]

    # create dataframe with aggregated predictions
    df = pd.DataFrame(preds)
    df = df.transpose().reset_index()
    df.columns = ["Sample ID", "Target"]
    return df
