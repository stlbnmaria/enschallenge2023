from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier


def logreg_estimator():
    estimator = LogisticRegression()
    return estimator


def xgb_estimator():
    estimator = XGBClassifier()
    return estimator


def catboost_estimator():
    estimator = CatBoostClassifier(colsample_bylevel=0.2)
    return estimator


def get_tabular_estimator(model: str):
    if model == "LogReg":
        return logreg_estimator()
    elif model == "RF":
        return RandomForestClassifier(n_jobs=4)
    elif model == "ET":
        return ExtraTreesClassifier()
    elif model == "XGB":
        return xgb_estimator()
    elif model == "Catboost":
        return catboost_estimator()
    else:
        print("The specified model is not valid")
