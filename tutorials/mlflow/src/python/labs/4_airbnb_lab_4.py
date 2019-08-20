'''
Databricks Learning Academy Lab -
'''
import mlflow.sklearn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

from lab_1 import RFRBaseModel

class RFFExperimentModel(RFRBaseModel):
    def __int__(self, params):

        RFRBaseModel.__init__(self, params)

    def mlflow_run(self, df, r_name="Basic RF Experiment"):
        print("Inside MLflow Run with experiment {} and parameters {}".format(r_name, params))


if __name__ == '__main__':
     params = {
        "n_estimators": 100,
        "max_depth": 5,
        "random_state": 42
     }

     rfr = RFFExperimentModel(params)
     df = rfr.load_data("data/airbnb-cleaned-mlflow.csv")
     rfr.mlflow_run(df, "Experiment 1 RF")

