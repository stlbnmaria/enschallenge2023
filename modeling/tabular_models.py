import os
from typing import Optional, Any

from ast import literal_eval
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier


def read_grid_tuning(path: str = "./modeling/") -> Optional[dict]:
    """
    This function loads the modeling inputs from a predefined excel input,
    e.g. the hyperparameter grid.
    """
    # read modeling input excel
    modeling_inputs = pd.read_excel(os.path.join(path, "modeling_inputs.xlsx"))
    try:
        # extract hyperparameter grid
        grid = modeling_inputs["grid"][0]
        grid = literal_eval(grid)
        print("Extracted grid successfully: ", grid)
    except:
        grid = None
        print("No hyperparameter grid is used")

    return grid


def get_tabular_estimator(model: str, n_jobs: int = 6) -> Any:
    """
    This function returns the corresponding estimator to the
    specified string argunent.
    """
    # define dict of potential estimators
    estimators = {
        "LogReg": LogisticRegression(),
        "RF": RandomForestClassifier(n_jobs=n_jobs, max_features=1),
        "ExtraTrees": ExtraTreesClassifier(n_jobs=n_jobs, max_features=1),
        "XGB": XGBClassifier(n_jobs=n_jobs),
        "Catboost": CatBoostClassifier(verbose=0),
        "LightGBM": LGBMClassifier(n_jobs=n_jobs),
        "SVC": SVC(probability=True),
    }

    # get estimator based on specified specified model
    try:
        return estimators[model]
    except:
        print("The specified model is not valid")
