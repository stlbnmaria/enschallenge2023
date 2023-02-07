import argparse
from pathlib import Path

import pandas as pd

from utils import store_submission
from training_functions import train_tabular

###############################################################################

parser = argparse.ArgumentParser(description = 'Main Script to Run Training')
parser.add_argument('--model', type = str, default = r"XGB", help = 'Type of model to run training')
parser.add_argument('--goal', type = str, default = r"test", help = 'Goal of the run (test/submission/tuning)')
args = parser.parse_args()  

###############################################################################

if __name__ == "__main__":

    input_args = vars(args)

    preds = train_tabular(input_args["model"]) 

    if input_args["goal"] == "submission":
        store_submission(preds, "test")
