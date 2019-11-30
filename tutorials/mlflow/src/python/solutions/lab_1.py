"""

Problem - part 1: We want to predict the gas consumption in millions of gallons in 48 of the US states
based on some key features. These features are petrol tax (in cents), per capital income (in US dollars),
paved highway (in miles), population of people with driving licences

Solution:

Since this is a regression problem where the value is a range of numbers, we can use the
common Random Forest Algorithm in Scikit-Learn. Most regression models are evaluated with
three standard evalution metrics: Mean Absolute Error(MAE); Mean Squared Error (MSE); and
Root Mean Squared Error (RSME), and r2.

This example is borrowed from the source below, modified and modularized for this tutorial
source: https://stackabuse.com/random-forest-algorithm-with-python-and-scikit-learn/

Aim of this Lab:

1. Understand MLflow Tracking API
2. How to use the MLflow Tracking API
3. Use the MLflow API to experiment several Runs
4. Interpret and observer runs via the MLflow UI

Some Resources:
https://mlflow.org/docs/latest/python_api/mlflow.html
https://www.saedsayad.com/decision_tree_reg.htm
https://towardsdatascience.com/understanding-random-forest-58381e0602d2
https://stackabuse.com/random-forest-algorithm-with-python-and-scikit-learn/
https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html
"""

import numpy as np
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
from lab_utils import Utils

class RFRModel():
    """
    General class for Scikit-learn RandomForestRegressor
    """
    # class wide variables common to all instances
    # keep track of cumulative estimators and rsme
    rsme = []
    estimators = []
    r2 = []

    def __init__(self, params={}):
        """
        Construtor for the RandomForestRegressor
        :param params: dictionary to RandomForestRegressor
        """
        self.rf = RandomForestRegressor(**params)
        self.params = params

    @classmethod
    def new_instance(cls, params={}):
        return cls(params)

    def model(self):
        """
        Return the model craeted
        :return: handle or instance of the RandomForestReqgressor
        """
        return self.rf

    def mlflow_run(self, df, r_name="Lab-1:RF Petrol Regression Experiment"):
        """
        This method trains, computes metrics, and logs all metrics, parameters,
        and artifacts for the current run
        :param df: pandas dataFrame
        :param r_name: Name of the experiment as logged by MLflow
        :return: MLflow Tuple (ExperimentID, runID)
        """

        with mlflow.start_run(run_name=r_name) as run:
            # get all rows and columns but the last column
            X = dataset.iloc[:, 0:4].values
            # get all the last columns, which is what we want to predict
            y = dataset.iloc[:, 4].values

            # create train and test data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

            # Feature Scaling
            sc = StandardScaler()
            X_train = sc.fit_transform(X_train)
            X_test = sc.transform(X_test)

            # train and predict
            self.rf.fit(X_train, y_train)
            y_pred = self.rf.predict(X_test)

            # Log model and params using the MLflow sklearn APIs
            mlflow.sklearn.log_model(self.rf, "random-forest-reg-model")
            mlflow.log_params(self.params)

            # compute  metrics
            mae = metrics.mean_absolute_error(y_test, y_pred)
            mse = metrics.mean_squared_error(y_test, y_pred)
            rsme = np.sqrt(mse)
            r2 = metrics.r2_score(y_test, y_pred)

            # Log metrics
            mlflow.log_metric("mae", mae)
            mlflow.log_metric("mse", mse)
            mlflow.log_metric("rsme", rsme)
            mlflow.log_metric("r2", r2)

            # update global class instance variable with values
            self.rsme.append(rsme)
            self.r2.append(r2)
            self.estimators.append(params["n_estimators"])

            # plot RSME graph and save as artifacts
            (fig, ax) = Utils.plot_graphs(rfr.estimators, rfr.rsme, "Random Forest Estimators", "Root Mean Square", "Root Mean Square vs Estimators")

            # get current run and experiment id
            runID = run.info.run_uuid
            experimentID = run.info.experiment_id

            # create temporary artifact file name and log artifact
            temp_file_name = Utils.get_temporary_directory_path("rsme_estimators-", ".png")
            temp_name = temp_file_name.name
            try:
                fig.savefig(temp_name)
                mlflow.log_artifact(temp_name, "rsme_estimators_plots")
            finally:
                temp_file_name.close()  # Delete the temp file

            # plot R2 graph and save as artifacts
            (fig_2, ax) = Utils.plot_graphs(rfr.estimators, rfr.r2, "Random Forest Estimators", "R2", "R2 vs Estimators")

            # create temporary artifact file name and log artifact
            temp_file_name = Utils.get_temporary_directory_path("r2-estimators-", ".png")
            temp_name = temp_file_name.name
            try:
                fig_2.savefig(temp_name)
                mlflow.log_artifact(temp_name, "r2_estimators_plots")
            finally:
                temp_file_name.close()  # Delete the temp file

            # print some data
            print("-" * 100)
            print("Inside MLflow Run with run_id {} and experiment_id {}".format(runID, experimentID))
            print("Estimator trees        :", self.params["n_estimators"])
            print("Estimator trees depth  :", self.params["max_depth"])
            print('Mean Absolute Error    :', mae)
            print('Mean Squared Error     :', mse)
            print('Root Mean Squared Error:', rsme)
            print('R2                     :', r2)

            return (experimentID, runID)

#
# TODO in Lab/Homework for Some Experimental runs
#
    # 1. Consult RandomForestRegressor documentation
    # 2. Change or add parameters, such as depth of the tree or random_state: 42 etc.
    # 3. Change or alter the range of runs and increments of n_estimators.
    # 4. Check in MLfow UI if the metrics are affected
    # challenge-1: create mean square error and r2 artifacts and save them for each run

if __name__ == '__main__':
    # load and print dataset
    dataset = Utils.load_data("data/petrol_consumption.csv")
    Utils.print_pandas_dataset(dataset)
    # iterate over several runs with different parameters, stepping up by 50
    # limiting to 300
    max_depth = 0
    for n in range (50, 350, 50):
        max_depth = max_depth + 2
        params = {"n_estimators": n, "max_depth": max_depth, "random_state": 42}
        rfr = RFRModel.new_instance(params)
        (experimentID, runID) = rfr.mlflow_run(dataset)
        print("MLflow Run completed with run_id {} and experiment_id {}".format(runID, experimentID))
        print("-" * 100)
