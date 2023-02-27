from catboost import CatBoostClassifier
from lce import LCEClassifier
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier


def get_tabular_estimator(model: str):
    # define dict of potential estimators
    estimators = {
        "LogReg": LogisticRegression(),
        "DT": DecisionTreeClassifier(),
        "RF": RandomForestClassifier(),
        "ExtraTrees": ExtraTreesClassifier(),
        "XGB": XGBClassifier(),
        "Catboost": CatBoostClassifier(verbose=0), 
        "LightGBM": LGBMClassifier(), 
        "LCE": LCEClassifier(),      
    }

    # get estimator based on specified specified model
    try:
        estimator = estimators[model]
    except:
        print("The specified model is not valid")
