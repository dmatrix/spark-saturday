import os
import shutil

from random import random, randint
import mlflow
from mlflow import log_metric, log_param, log_artifacts
from mlflow.tracking import MlflowClient

if __name__ == "__main__":

    # set the tracking server to be Databricks Community Edition
    # set the experiment name; if name does not exist, MLflow will
    # create one for you
    mlflow.set_tracking_uri("databricks")
    mlflow.set_experiment("/Users/jules.damji@gmail.com/Jules_CE_Test")
    print("Running experiment_ce.py")
    print("Tracking on https://community.cloud.databricks.com")
    mlflow.start_run(run_name="CE_TEST")
    log_param("param-1", randint(0, 100))

    log_metric("metric-1", random())
    log_metric("metric-2", random() + 1)
    log_metric("metric-3", random() + 2)

    if not os.path.exists("outputs"):
        os.makedirs("outputs")
    with open("outputs/test.txt", "w") as f:
        f.write("Looks, like I logged on the Community Edition!")

    log_artifacts("outputs")
    shutil.rmtree('outputs')
    mlflow.end_run()
    #
    # try model registry on CE: not yet supported
    #
    """
    cltn = MlflowClient()
    cltn.create_registered_model("test_model_registry_on_CE")
    """



