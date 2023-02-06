from xgboost import XGBClassifier


def xgb_estimator():
    estimator = XGBClassifier()
    return estimator

def get_tabular_estimator(model: str):
    if model == "XGB":
        return xgb_estimator()
    else:
        print("The specified model is not valid")
