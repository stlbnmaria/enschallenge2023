from xgboost import XGBClassifier


def xgb_estimator():
    estimator = XGBClassifier(subsample=0.1, colsample_bylevel=0.1)
    return estimator

def get_tabular_estimator(model: str):
    if model == "XGB":
        return xgb_estimator()
    else:
        print("The specified model is not valid")
