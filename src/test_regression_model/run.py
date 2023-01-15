#!/usr/bin/env python
import argparse
import itertools
import logging
import json
import pandas as pd
import wandb
import mlflow.sklearn
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, plot_confusion_matrix
from sklearn.metrics import mean_absolute_error

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def go(args):

    run = wandb.init(job_type="test")
    
    logger.info("Downloading and reading test artifact")
    test_data_path = run.use_artifact(args.test_data).file()
    df = pd.read_csv(test_data_path, low_memory=False)

    # Extract the target from the features
    logger.info("Extracting target from dataframe")
    X_test = df.copy()
    y_test = X_test.pop("price")

    logger.info("Downloading and reading the exported model")
    model_export_path = run.use_artifact(args.model_export).download()

    sk_pipe  =  mlflow.sklearn.load_model(model_export_path)
    processed_features = list(itertools.chain.from_iterable([x[2] for x in sk_pipe['Preprocessor'].transformers]))

    # Compute r2 and MAE
    logger.info("Scoring")
    r_squared = sk_pipe.score(X_test[processed_features], y_test)

    y_pred = sk_pipe.predict(X_test[processed_features])
    mae = mean_absolute_error(y_test, y_pred)

    logger.info(f"Score: {r_squared}")
    logger.info(f"MAE: {mae}")

    ######################################
    # Here we save r_squared under the "r2" key
    run.summary['r2'] = r_squared
    # Now log the variable "mae" under the key "mae".
    run.summary["mae"] = mae

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Test the provided model on the test artifact",
        fromfile_prefix_chars="@",
    )

    parser.add_argument(
        "--model_export",
        type=str,
        help="Fully-qualified artifact name for the exported model to evaluate",
        required=True,
    )

    parser.add_argument(
        "--test_data",
        type=str,
        help="Fully-qualified artifact name for the test data",
        required=True,
    )

    args = parser.parse_args()

    go(args)
