from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression


def logreg_estimator():
    estimator = LogisticRegression()
    return estimator


def xgb_estimator():
    estimator = XGBClassifier(colsample_bylevel=0.2)
    return estimator

def catboost_estimator():
    estimator = CatBoostClassifier(colsample_bylevel=0.2)
    return estimator

def get_tabular_estimator(model: str):
    if model == "LogReg":
        return logreg_estimator()
    elif model == "XGB":
        return xgb_estimator()
    elif model == "Catboost":
        return catboost_estimator()
    else:
        print("The specified model is not valid")
