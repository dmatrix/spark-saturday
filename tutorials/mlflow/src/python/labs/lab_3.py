
'''
Databricks Learning Academy Lab -
'''
import mlflow.sklearn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

class RFRBaseModel():

    def __init__(self, params={}):
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

            # Log model
            mlflow.sklearn.log_model(self.rf, "random-forest-model")

            # Create metrics
            mse = mean_squared_error(y_test, predictions)
            print("  mse: {}".format(mse))

            # Log metrics
            mlflow.log_metric("mse", mse)

            runID = run.info.run_uuid
            experimentID = run.info.experiment_id

            print("Inside MLflow Run with run_id {} and experiment_id {}".format(runID, experimentID))

if __name__ == '__main__':
     rfr = RFRBaseModel()
     df = rfr.load_data("data/airbnb-cleaned-mlflow.csv")
     rfr.mlflow_run(df)
