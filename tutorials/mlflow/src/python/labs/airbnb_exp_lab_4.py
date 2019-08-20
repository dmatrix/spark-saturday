'''
Databricks Learning Academy Lab -

While iterating or build models, data scientists will often create a base line model to see how the model performs.
And then iterate with experiments, changing or altering parameters to ascertain how the new parameters or
hyper-parameters move the metrics closer to their confidence level.

This is our base line model using RandomForestRegressor model to predict AirBnb house prices in SF.
Given 22 features can we predict what the next house price will be?

We will compute standard evalution metrics and log them.

Aim of this module is:

1. Introduce tracking ML experiments in MLflow
2. Log a base experiment and explore the results in the UI
3. Record parameters, metrics, and a model

Some Resources:
https://mlflow.org/docs/latest/python_api/mlflow.html
https://www.saedsayad.com/decision_tree_reg.htm
https://stackabuse.com/random-forest-algorithm-with-python-and-scikit-learn/
https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html

'''
import mlflow.sklearn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from airbnb_base_lab_3 import RFRBaseModel
from lab_utils import load_data

class RFFExperimentModel(RFRBaseModel):
    def __int__(self, params):

        RFRBaseModel.__init__(self, params)

    def mlflow_run(self, df, r_name="Basic RF Experiment"):
        print("Inside MLflow Run with experiment {} and parameters {}".format(r_name, params))


if __name__ == '__main__':
     params_100_trees = {
        "n_estimators": 100,
        "max_depth": 5,
        "random_state": 42 }
     params_1000_trees = {
        "n_estimators": 1000,
        "max_depth": 10,
        "random_state": 42 }
     param_3000_trees = {
        "n_estimators": 3000,
        "max_depth": 20,
        "random_state": 42 }

     dataset = load_data("data/airbnb-cleaned-mlflow.csv")
     for params in [params_100_trees,params_1000_trees, param_3000_trees]:
         rfr = RFFExperimentModel(params)
         experiment = "Experiment with " + str(params['n_estimators'])
         rfr.mlflow_run(dataset, experiment)

