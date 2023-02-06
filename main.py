import argparse

from utils import store_submission
from modeling.tabular_models import get_tabular_estimator

###############################################################################

parser = argparse.ArgumentParser(description = 'Main Script to Run Training')
parser.add_argument('--model', type = str, default = r"XGB", help = 'Type of Model to Run Training')
args = parser.parse_args()  

###############################################################################

if __name__ == "__main__":

    model = vars(args)["model"]
    estimator = get_tabular_estimator(model)
    preds = []    
    # store_submission(preds, "test")
