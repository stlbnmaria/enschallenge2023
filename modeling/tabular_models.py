from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier


def get_tabular_estimator(model: str):
    # define dict of potential estimators
    estimators = {
        "LogReg": LogisticRegression(),
        "DT": DecisionTreeClassifier(),
        "RF": RandomForestClassifier(n_jobs=6),
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
