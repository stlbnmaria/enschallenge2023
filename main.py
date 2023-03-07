import argparse

from utils import store_submission
from training_functions import train_tabular, tuning_moco

###############################################################################

parser = argparse.ArgumentParser(description="Main Script to Run Training")
parser.add_argument(
    "--model", type=str, default=r"XGB", help="Type of model to run training"
)
parser.add_argument(
    "--goal", type=str, default=r"test", help="Goal of the run (test/submission/tuning)"
)
parser.add_argument(
    "--subname", type=str, default=r"test", help="Name of the submission"
)
parser.add_argument(
    "--aggregation",
    type=str,
    default=r"mean",
    help="Aggreagtion of predictions in test & training",
)
parser.add_argument(
    "--parallel",
    type=int,
    default=6,
    help="Number of jobs to run in parallel for fit of models",
)
args = parser.parse_args()

###############################################################################

if __name__ == "__main__":

    input_args = vars(args)

    if input_args["goal"] == "test":
        preds = train_tabular(input_args["model"], input_args["aggregation"])
    elif input_args["goal"] == "submission":
        preds = train_tabular(input_args["model"], input_args["aggregation"])
        store_submission(preds, input_args["subname"])
    elif input_args["goal"] == "tuning":
        tuning_moco(model=input_args["model"], agg_by=input_args["aggregation"])
