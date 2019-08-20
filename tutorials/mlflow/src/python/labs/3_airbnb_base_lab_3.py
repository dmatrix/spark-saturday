
'''
Databricks Learning Academy Lab
'''
import mlflow.sklearn
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
from lab_utils import load_data, print_pandas_dataset


class RFRBaseModel():

    def __init__(self, params={}):
        self.params = params
        self.rf = RandomForestRegressor(**params)

    def model(self):
        return self.rf

    def load_data(self, path):
        self.df = pd.read_csv(path)
        return self.df

    def mlflow_run(self, df, r_name="Basic RF Experiment"):
        with mlflow.start_run(run_name=r_name) as run:
            X_train, X_test, y_train, y_test = train_test_split(df.drop(["price"], axis=1), df[["price"]].values.ravel(), random_state=42)
            self.rf.fit(X_train, y_train)
            predictions = self.rf.predict(X_test)

            # Log model and parameters
            mlflow.sklearn.log_model(self.rf, "random-forest-model")
            mlflow.log_params(self.params)

            # Create metrics
            mae = metrics.mean_absolute_error(y_test, predictions)
            mse = metrics.mean_squared_error(y_test, predictions)
            rsme = np.sqrt(mse)
            r2 = metrics.r2_score(y_test, predictions)

            # Log metrics
            mlflow.log_metric("mae", mae)
            mlflow.log_metric("mse", mse)
            mlflow.log_metric("rsme", rsme)
            mlflow.log_metric("r2", r2)

            runID = run.info.run_uuid
            experimentID = run.info.experiment_id

            # print some data
            print("Inside MLflow Run with run_id {} and experiment_id {}".format(runID, experimentID))
            print("Estimator trees        :", self.params["n_estimators"])
            print('Mean Absolute Error    :', mae)
            print('Mean Squared Error     :', mse)
            print('Root Mean Squared Error:', rsme)
            print('R2                     :', r2)
            print("-" * 100)

if __name__ == '__main__':
    # load and print dataset
    dataset = load_data("data/airbnb-cleaned-mlflow.csv")
    print_pandas_dataset(dataset)
    # create a base model parameters
    params = {"n_estimators": 100, "random_state": 42 }
    rfr = RFRBaseModel(params)
    rfr.mlflow_run(dataset)
