"""
source: Databricks Learning Academy Lab

Refactored code to modularize it

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
https://towardsdatascience.com/understanding-random-forest-58381e0602d2
https://github.com/MangoTheCat/Modelling-Airbnb-Prices
"""

import mlflow.sklearn
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
from lab_utils import Utils

class RFRBaseModel():

    def __init__(self, params={}):
        """
        Construtor for the RandomForestRegressor
        :param params: dictionary to RandomForestRegressor
        """
        self.params = params
        self.rf = RandomForestRegressor(**params)

    @classmethod
    def new_instance(cls, params={}):
        return cls(params)

    def model(self):
        """
        Getter for the model
        :return: return the model
        """
        return self.rf

    def mlflow_run(self, df, r_name="Lab-3: Baseline RF Model"):
        """
        This method trains, computes metrics, and logs all metrics, parameters,
        and artifacts for the current run
        :param df: pandas dataFrame
        :param r_name: Name of the experiment as logged by MLflow
        :return: Tuple of MLflow experimentID, runID
        """
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
            print("-" * 100)
            print("Inside MLflow {} Run with run_id {} and experiment_id {}".format(r_name, runID, experimentID))
            print("Estimator trees        :", self.params["n_estimators"])
            print('Mean Absolute Error    :', mae)
            print('Mean Squared Error     :', mse)
            print('Root Mean Squared Error:', rsme)
            print('R2                     :', r2)

            return (experimentID, runID)
#
# TODO in Lab/Homework for Some Experimental runs
#
    # 1. Consult RandomForest documentation
    # 2. Run the baseline model
    # 3. Check in MLflow UI for parameters, metrics, and artifacts

if __name__ == '__main__':
    # load and print dataset
    dataset = Utils.load_data("data/airbnb-cleaned-mlflow.csv")
    Utils.print_pandas_dataset(dataset)
    #
    # create a base line model parameters
    # this is our benchmark model to compare experimental results with
    #
    params = {"n_estimators": 100, "max_depth": 3, "random_state": 0}
    rfr = RFRBaseModel.new_instance(params)
    (experimentID, runID) = rfr.mlflow_run(dataset)
    print("MLflow completed with run_id {} and experiment_id {}".format(runID, experimentID))
    print("-" * 100)
