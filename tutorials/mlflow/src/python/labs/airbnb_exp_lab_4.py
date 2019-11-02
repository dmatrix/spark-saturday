"""
Databricks Learning Academy Lab -

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
https://towardsdatascience.com/understanding-random-forest-58381e0602d2
https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html
https://towardsdatascience.com/explaining-feature-importance-by-example-of-a-random-forest-d9166011959e
https://seaborn.pydata.org/tutorial/regression.html
"""

import mlflow.sklearn
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn import metrics
from  mlflow.tracking import MlflowClient

from airbnb_base_lab_3 import RFRBaseModel
from lab_utils import Utils


class RFFExperimentModel(RFRBaseModel):
    """
    Constructor for the Experimental RandomForestRegressor.
    """
    def __int__(self, params):
        """
        Call the superclass initializer
        :param params: parameters for the RandomForestRegressor instance
        :return: None
        """
        super(RFRBaseModel, self).__init__(params)

    def mlflow_run(self, df, r_name="Lab-4:RF Experiment Model"):
        """
        Override the base class mlflow_run for this epxerimental runs
        This method trains the model, evaluates, computes the metrics, logs
        all the relevant metrics, artifacts, and models.
        :param df: pandas dataFrame
        :param r_name: name of the experiment run
        :return:  MLflow Tuple (ExperimentID, runID)
        """

        with mlflow.start_run(run_name=r_name) as run:
            X_train, X_test, y_train, y_test = train_test_split(df.drop(["price"], axis=1), df[["price"]].values.ravel(), random_state=42)
            self.rf.fit(X_train, y_train)
            predictions = self.rf.predict(X_test)

            # Log model and parameters
            mlflow.sklearn.log_model(self.rf, "random-forest-model")

            # Note we are logging as a dictionary of all params instead of logging each parameter
            mlflow.log_params(self.params)

            # Log params
            #[mlflow.log_param(param, value) for param, value in self.params.items()]

            # Create metrics
            mse = metrics.mean_squared_error(y_test, predictions)
            rmse = np.sqrt(mse)
            mae = metrics.mean_absolute_error(y_test, predictions)
            r2 = metrics.r2_score(y_test, predictions)

            # Log metrics
            mlflow.log_metric("mse", mse)
            mlflow.log_metric("mae", mae)
            mlflow.log_metric("rsme", rmse)
            mlflow.log_metric("r2", r2)

            # get experimentalID and runID
            runID = run.info.run_uuid
            experimentID = run.info.experiment_id

            # Create feature importance and save them as artifact
            # This allows us to remove least important features from the dataset
            # with each iteration if they don't have any effect on the predictive power of
            # the prediction.
            importance = pd.DataFrame(list(zip(df.columns, self.rf.feature_importances_)),
                                      columns=["Feature", "Importance"]
                                      ).sort_values("Importance", ascending=False)

            # Log importance file as feature artifact
            temp_file_name = Utils.get_temporary_directory_path("feature-importance-", ".csv")
            temp_name = temp_file_name.name
            try:
                importance.to_csv(temp_name, index=False)
                mlflow.log_artifact(temp_name, "feature-importance-files")
            finally:
                temp_file_name.close()  # Delete the temp file

            # Create residual plots and image directory
            # Residuals R = observed value - predicted value
            (plt, fig, ax) = Utils.plot_residual_graphs(predictions, y_test, "Predicted values for Price ($)","Residual" , "Residual Plot")

            # Log residuals images
            temp_file_name = Utils.get_temporary_directory_path("residuals-", ".png")
            temp_name = temp_file_name.name
            try:
                fig.savefig(temp_name)
                mlflow.log_artifact(temp_name, "residuals-plots")
            finally:
                temp_file_name.close()  # Delete the temp file
            
            print("-" * 100)
            print("Inside MLflow {} Run with run_id {} and experiment_id {}".format(r_name, runID, experimentID))
            print("  mse: {}".format(mse))
            print(" rmse: {}".format(rmse))
            print("  mae: {}".format(mae))
            print("  R2 : {}".format(r2))

            return (experimentID, runID)
#
# Lab/Homework for Some Experimental runs
#
    # 1. Consult RandomForestRegressor documentation
    # 2. Change or add parameters, such as depth of the tree or random_state: 42 etc.
    # 3. Change or alter the range of runs and increments of n_estimators
    # 4. Check in MLfow UI if the metrics are affected

if __name__ == '__main__':
    # TODO add more parameters to the list
    # create four experiments with different parameters
    # run these different experiments, each with its own instance of model with the supplied parameters.
    # add more parameters to this dictionary list here
    params_list = [
        {"n_estimators": 200,"max_depth": 6, "random_state": 42}
    ]
    # load the data
    dataset = Utils.load_data("data/airbnb-cleaned-mlflow.csv")

    # run these experiments, each with its own instance of model with the supplied parameters.
    for params in params_list:
        rfr = RFFExperimentModel.new_instance(params)
        experiment = "Experiment with {} trees".format(params['n_estimators'])
        (experimentID, runID) = rfr.mlflow_run(dataset, experiment)
        print("MLflow Run completed with run_id {} and experiment_id {}".format(runID, experimentID))
        print("-" * 100)

    # Use MLflowClient API to query programmatically any previous run info under an experiment ID
    # consult https://mlflow.org/docs/latest/python_api/mlflow.tracking.html
    client = MlflowClient()
    run_list = client.list_run_infos(experimentID)
    [print(rinfo) for rinfo in run_list]

