import sys
from pathlib import Path
import pandas as pd
from xgboost import XGBClassifier

sys.path.append("..")
import modeling


def main():
    data_path = Path("../storage")
    output = pd.read_csv(data_path / "train_output.csv")
    md_train = pd.read_csv(data_path / "supplementary_data" / "train_metadata.csv")
    output_md = md_train.merge(output, on="Sample ID")

    (
        X_train,
        y_train,
        patients_unique,
        y_unique,
        patients_train,
    ) = modeling.load_train_data(output_md)
    lrs = modeling.train_mocov_features(
        XGBClassifier(), X_train, y_train, patients_unique, y_unique, patients_train
    )
    preds = modeling.predict_cv_classifiers(lrs)
    modeling.store_submission(preds, "test")


if __name__ == "__main__":
    main()
