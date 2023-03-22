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
        # stack data for duplicate selection
        df = pd.DataFrame({"patient": patients_train, 
                            "zoom": coords, 
                            "samples": samples_train})
        df = df.loc[::1000, :]
        # specify manual prod list of samples
        drop_list = ['ID_014.npy', 'ID_028.npy', 'ID_052.npy', 
                     'ID_117.npy','ID_130.npy', 'ID_132.npy', 
                     'ID_233.npy', 'ID_474.npy', 'ID_260.npy', 
                     'ID_296.npy', 'ID_324.npy', 'ID_355.npy', 
                     'ID_440.npy', 'ID_455.npy',
                     'ID_074.npy', 'ID_083.npy', 'ID_102.npy', 
                     'ID_094.npy', 'ID_243.npy', 'ID_055.npy', 
                     'ID_257.npy', 'ID_272.npy', 'ID_274.npy', 
                     'ID_278.npy', 'ID_279.npy', 'ID_316.npy', 
                     'ID_328.npy', 'ID_334.npy', 'ID_099.npy', 
                     'ID_358.npy', 'ID_371.npy', 'ID_411.npy', 
                     'ID_437.npy', 'ID_460.npy', 'ID_461.npy', 
                     'ID_465.npy', 'ID_481.npy', 
                     'ID_484.npy', 'ID_491.npy']
        # extract index for samples to keep and replicate the original list for tiles
        df = df[~df.samples.isin(drop_list)]
        idx = df.sort_index().index
        idx = np.array([y for i in idx for y in list(range(i, i + 1000))])

        # filter data according to index - leaving 305 samples
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
        # # scale the feature values for each center seperately
        # X_train = np.empty([0, 2048])
        scale_dict = {}
        # if tile_averaging == "pos_avg":
        #     # change zeros to nan so that they are ignored during scaling
        #     feat = np.where(feat==0, np.nan, feat)
        # for zoom in np.unique(coords):
        #     scaler = {"MinMax": MinMaxScaler(), "Standard": StandardScaler()}[scaling]
        #     X_train = np.vstack(
        #         [X_train, scaler.fit_transform(feat[coords == zoom])]
        #     )
        #     scale_dict[zoom] = scaler

        # # conditions for the three centers to reorder the other arrays
        # c1 = coords == 15.
        # c2 = coords == 16.
        # c5 = coords == 17.

        # # reorder y, patients, samples and centers arrays
        # y_train = np.hstack([y_train[c1], y_train[c2], y_train[c5]])
        # patients_train = np.hstack(
        #     [patients_train[c1], patients_train[c2], patients_train[c5]]
        # )
        # samples_train = np.hstack(
        #     [samples_train[c1], samples_train[c2], samples_train[c5]]
        # )
        # coords = np.hstack([coords[c1], coords[c2], coords[c5]])
        # centers_train = np.hstack([centers_train[c1], centers_train[c2], centers_train[c5]])
        # X_train = np.column_stack((coords, X_train))

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

    imp_idx = np.array(pd.read_csv("./modeling/temp.csv").iloc[:,0]) 

    return (
        X_train[:, imp_idx],
        y_train,
        patients_train,
        samples_train,
        centers_train,
        scale_dict
    )


def load_mocov_test_data(
    data_path=Path("./storage/"), tile_averaging: str = None, scaling: str = None, onehot_zoom: bool = False, scale_dict: dict = None
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
        # # scale the feature values for each center seperately
        # X_test = np.empty([0, 2048])
        # if tile_averaging == "pos_avg":
        #     # change zeros to nan so that they are ignored during scaling
        #     feat = np.where(feat==0, np.nan, feat)
        # for zoom in np.unique(coords):
        #     if zoom == 14.:
        #         scaler = {"MinMax": MinMaxScaler(), "Standard": StandardScaler()}[scaling]
        #         X_test = np.vstack(
        #             [X_test, scaler.fit_transform(feat[coords == zoom])]
        #         )
        #     else: 
        #         X_test = np.vstack(
        #             [X_test, scale_dict[zoom].transform(feat[coords == zoom])]
        #         )

        # # conditions for the three centers to reorder the other arrays
        # c3 = coords == 14.
        # c4 = coords == 15.
        # c5 = coords == 16.
        # c6 = coords == 17.

        # # reorder patients, samples and centers arrays
        # patients_test = np.hstack([patients_test[c3], patients_test[c4], patients_test[c5], patients_test[c6]])
        # samples_test = np.hstack([samples_test[c3], samples_test[c4], samples_test[c5], samples_test[c6]])
        # coords = np.hstack([coords[c3], coords[c4], coords[c5], coords[c6]])
        # centers_test = np.hstack([centers_test[c3], centers_test[c4], centers_test[c5], centers_test[c6]])
        # X_test = np.column_stack((coords, X_test))

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

    imp_idx = np.array(pd.read_csv("./modeling/temp.csv").iloc[:,0]) 

    return X_test[:, imp_idx], patients_test, samples_test, centers_test


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
