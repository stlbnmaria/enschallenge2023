import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm


def write_mocov_csv(data_path=Path("../storage")):
    """
    This function loads the MoCov features for training and stores one
    compressed numpy file incl. all tile values.
    """
    output = pd.read_csv(data_path / "train_output.csv")
    md_train = pd.read_csv(data_path / "supplementary_data" / "train_metadata.csv")
    metadata = md_train.merge(output, on="Sample ID")

    train_features_dir = data_path / "train_input" / "moco_features"

    X_train = np.empty([0, 2048])
    y_train = np.empty([])
    centers_train = np.empty([])
    patients_train = np.empty([])
    samples_train = np.empty([])

    for sample, label, center, patient in tqdm(
        metadata[["Sample ID", "Target", "Center ID", "Patient ID"]].values
    ):
        # load the coordinates and features (1000, 3+2048)
        _features = np.load(train_features_dir / sample)
        # get coordinates (zoom level, tile x-coord on the slide, tile y-coord on the slide)
        # and the MoCo V2 features
        _, features = _features[:, :3], _features[:, 3:]  # Ks

        n = features.shape[0]
        X_train = np.vstack((X_train, features))
        y_train = np.hstack((y_train, [label] * n))
        centers_train = np.hstack((centers_train, [center] * n))
        patients_train = np.hstack((patients_train, [patient] * n))
        samples_train = np.hstack((samples_train, [sample] * n))

    # store csv
    output = np.vstack((y_train[1:], patients_train[1:], samples_train[1:], centers_train[1:])).T
    np.savez_compressed(
        data_path / "train_input" / "mocov_features_train",
        metadata=output,
        features=X_train,
    )

    print("------------------- File saved successfully -------------------")


if __name__ == "__main__":
    write_mocov_csv()
