from pathlib import Path
import time
from tqdm import tqdm

import numpy as np
import pandas as pd


def load_avg_test_mocov_features():
    data_path = Path("./storage/")
    test_features_dir = data_path / "test_input" / "moco_features"

    md_test = pd.read_csv(data_path / "supplementary_data" / "test_metadata.csv")

    X_test = []

    for sample in tqdm(md_test["Sample ID"].values):
        _features = np.load(test_features_dir / sample)
        _, features = _features[:, :3], _features[:, 3:]
        X_test.append(np.mean(features, axis=0))

    return np.array(X_test)


def store_submission(test_data, preds, sub_name: str):
    submission = pd.DataFrame(
        {"Sample ID": test_data["Sample ID"].values, "Target": preds}
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
        Path("./submissions") / f"{timestamp}_{sub_name}_output.csv", index=None
    )
