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
        "LogReg": LogisticRegression(class_weight={0:.35, 1:.65}),
        "DT": DecisionTreeClassifier(),
        "RF": RandomForestClassifier(max_features=1, n_jobs=6, class_weight={0:.1, 1:.9}),
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
