import argparse
from pathlib import Path

import pandas as pd

from utils import store_submission
from training_functions import load_train_data, train_mocov_features, predict_cv_classifiers
from modeling.tabular_models import get_tabular_estimator

###############################################################################

parser = argparse.ArgumentParser(description = 'Main Script to Run Training')
parser.add_argument('--model', type = str, default = r"XGB", help = 'Type of Model to Run Training')
args = parser.parse_args()  

###############################################################################

if __name__ == "__main__":

    model = vars(args)["model"]
    estimator = get_tabular_estimator(model)

    data_path = Path("./storage")
    output = pd.read_csv(data_path / "train_output.csv")
    md_train = pd.read_csv(data_path / "supplementary_data" / "train_metadata.csv")
    output_md = md_train.merge(output, on="Sample ID")

    (
        X_train,
        y_train,
        patients_unique,
        y_unique,
        patients_train,
    ) = load_train_data(output_md)
    lrs = train_mocov_features(
        estimator, X_train, y_train, patients_unique, y_unique, patients_train
    )
    preds = predict_cv_classifiers(lrs)  
    store_submission(preds, "test")
