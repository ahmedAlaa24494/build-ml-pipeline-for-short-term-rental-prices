#!/usr/bin/env python
"""
Download from W&B the raw dataset and apply some basic data cleaning, exporting the result to a new artifact
"""
import os
import argparse
import logging
import wandb
import pandas as pd


logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def go(args):

    run = wandb.init(job_type="basic_cleaning")
    run.config.update(args)

    # Download input artifact. This will also log that this script is using this
    # particular version of the artifact
    artifact_local_path = run.use_artifact(args.input_artifact).file('artifact')

    ######################
    # YOUR CODE HERE     #
    ######################
    df = pd.read_csv(artifact_local_path)

    logger.info("Drop outliers...")
    idx = df['price'].between(args.min_price, args.max_price)
    df = df[idx].copy()
    # Convert last_review to datetime
    logger.info("Convert last_review date from str to datetime")
    df['last_review'] = pd.to_datetime(df['last_review'])

    #Fill NA values
    df['last_review'] = df['last_review'].fillna('')
    df['name'] = df['name'].fillna('')
    df['host_name'] = df['host_name'].fillna('')
    df['reviews_per_month'] = df['reviews_per_month'].fillna(0)


    filename = "clean_sample.csv"
    df.to_csv(filename, index=False)

    artifact = wandb.Artifact(
        name=args.output_artifact,
        type=args.output_type,
        description=args.output_description,
    )
    artifact.add_file(filename)

    logger.info("Logging artifact")
    run.log_artifact(artifact)

    os.remove(filename)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="A very basic data cleaning")


    parser.add_argument(
        "--input_artifact", 
        type=str,
        help="WANDB artifact uri",
        required=True
    )

    parser.add_argument(
        "--output_artifact", 
        type=str,
        help="Name of the output artifact",
        required=True
    )

    parser.add_argument(
        "--output_type", 
        type=str,
        help="Type of the output artifact according to the pipeline",
        required=True
    )

    parser.add_argument(
        "--output_description", 
        type=str,
        help="The description of the output",
        required=True
    )

    parser.add_argument(
        "--min_price", 
        type=float,
        help="the exprected value for the minimum charge, based on subject matter expret point of view",
        default= 10.0 ,
        required=False
    )

    parser.add_argument(
        "--max_price", 
        type=float,
        help="the exprected value for the maximum charge, based on subject matter expret point of view",
        default = 350.0,
        required=False
    )


    args = parser.parse_args()

    go(args)
