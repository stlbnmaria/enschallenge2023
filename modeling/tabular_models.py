import os

from ast import literal_eval
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier


def read_grid_tuning(path: str = "./modeling/"):
    """
    This function loads the modeling inputs from a predefined excel input,
    inlc. the feature engineering selection inputs and
    hyperparameter grid.
    """
    modeling_inputs = pd.read_excel(os.path.join(path, "modeling_inputs.xlsx"))
    # extract hyperparameter grid
    grid = modeling_inputs["grid"][0]
    try:
        grid = literal_eval(grid)
    except:
        grid = None
    
    return grid


def get_tabular_estimator(model: str):
    # define dict of potential estimators
    estimators = {
        "LogReg": LogisticRegression(solver="newton-cholesky"),
        "DT": DecisionTreeClassifier(),
        "RF": RandomForestClassifier(n_jobs=6, n_estimators=300),
        "ExtraTrees": ExtraTreesClassifier(n_jobs=6),
        "XGB": XGBClassifier(),
        "Catboost": CatBoostClassifier(verbose=0), 
        "LightGBM": LGBMClassifier(),  
        "MLP": MLPClassifier(),
    }

    # get estimator based on specified specified model
    try:
        return estimators[model]
    except:
        print("The specified model is not valid")
