from xgboost import XGBClassifier
from catboost import CatBoostClassifier


def xgb_estimator():
    estimator = XGBClassifier(colsample_bylevel=0.2)
    return estimator

def catboost_estimator():
    estimator = CatBoostClassifier(colsample_bylevel=0.2)
    return estimator

def get_tabular_estimator(model: str):
    if model == "XGB":
        return xgb_estimator()
    elif model == "Catboost":
        return catboost_estimator()
    else:
        print("The specified model is not valid")
