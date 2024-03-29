import argparse
from ast import literal_eval

from inference import store_submission, train_for_submission
from training_functions import train_tabular, tuning_moco, stacking_estimators

###############################################################################

parser = argparse.ArgumentParser(description="Main script to run training")
parser.add_argument(
    "--model", type=str, default=r"XGB", help="Type of model to run training"
)
parser.add_argument(
    "--goal",
    type=str,
    default=r"test",
    help="Goal of the run (test/submission/tuning/stacking)",
)
parser.add_argument(
    "--subname", type=str, default=r"test", help="Name of the submission"
)
parser.add_argument(
    "--tile_avg",
    type=str,
    default=None,
    help="Tile averaging means no weakly supervision (None/pos_avg/avg)",
)
parser.add_argument(
    "--scaling",
    type=str,
    default=None,
    help="Scaling MoCo features for each center (MinMax/Standard/None)",
)
parser.add_argument(
    "--feat_select",
    type=bool,
    default=True,
    help="Indicate whether to select features",
)
parser.add_argument(
    "--drop",
    type=bool,
    default=True,
    help="Indicate whether to drop duplicates",
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
        train_tabular(
            model=input_args["model"],
            agg_by=input_args["aggregation"],
            tile_avg=input_args["tile_avg"],
            scaling=input_args["scaling"],
            drop=input_args["drop"],
            feat_select=input_args["feat_select"],
            n_jobs=input_args["parallel"],
        )
    elif input_args["goal"] == "submission":
        try:
            estim = literal_eval(input_args["model"])
            assert isinstance(estim, list)
        except:
            estim = input_args["model"]

        preds = train_for_submission(
            model=estim,
            agg_by=input_args["aggregation"],
            tile_avg=input_args["tile_avg"],
            scaling=input_args["scaling"],
            drop=input_args["drop"],
            feat_select=input_args["feat_select"],
            n_jobs=input_args["parallel"],
        )
        store_submission(preds=preds, sub_name=input_args["subname"])
    elif input_args["goal"] == "tuning":
        tuning_moco(
            model=input_args["model"],
            agg_by=input_args["aggregation"],
            tile_avg=input_args["tile_avg"],
            scaling=input_args["scaling"],
            drop=input_args["drop"],
            feat_select=input_args["feat_select"],
            n_jobs=input_args["parallel"],
            file_name=input_args["subname"],
        )
    elif input_args["goal"] == "stacking":
        models = literal_eval(input_args["model"])
        assert isinstance(models, list)
        stacking_estimators(
            models=models,
            tile_avg=input_args["tile_avg"],
            scaling=input_args["scaling"],
            drop=input_args["drop"],
            feat_select=input_args["feat_select"],
            n_jobs=input_args["parallel"],
        )
