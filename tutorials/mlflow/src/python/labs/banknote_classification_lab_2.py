"""

Problem - part 2: Given a set of features or attributes of a bank note, can we predict whether it's authentic or fake
Four attributes contribute to this classification:
1. variance or transformed image
2. skeweness
3. entropy
4. cutoosis

Solution:

We are going to use Random Forest Classification to make the prediction, and measure on the accuracy.
The closer to 1.0 is the accuracy the better is our confidence in its prediction.

This example is borrowed from the source below, modified and modularized for this tutorial
source: https://stackabuse.com/random-forest-algorithm-with-python-and-scikit-learn/
source:https://archive.ics.uci.edu/ml/datasets/banknote+authentication

Aim of this Lab:

1. Understand MLflow Tracking API
2. How to use the MLflow Tracking API
3. Use the MLflow API to experiment several Runs
4. Interpret and observer runs via the MLflow UI

Some resources:
https://mlflow.org/docs/latest/python_api/mlflow.html
https://devopedia.org/confusion-matrix
https://stackabuse.com/random-forest-algorithm-with-python-and-scikit-learn/
"""

import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score
from lab_utils import Utils

class RFCModel():

    def __init__(self, params={}):
        """
        Constructor for RandomForestClassifier
        :param params: parameters for the constructor such as no of estimators, depth of the tree, random_state etc
        """
        self.rf = RandomForestClassifier(**params)
        self.params = params

    @classmethod
    def new_instance(cls, params={}):
        return cls(params)

    def model(self):
        """
        Fetch the model
        :return: return the model
        """
        return self.rf

    def mlflow_run(self, df, r_name="Lab-2:RF Bank Note Classification Experiment"):
        """
        This method trains, computes metrics, and logs all metrics, parameters,
        and artifacts for the current run
        :param df: pandas dataFrame
        :param r_name: Name of the experiment as logged by MLflow
        :return: MLflow Tuple (ExperimentID, runID)
        """

        with mlflow.start_run(run_name=r_name) as run:
            # get all attributes
            X = df.iloc[:, 0:4].values
            # get all the last columns, which is what we want to predict, our values
            y = df.iloc[:, 4].values

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
            mlflow.sklearn.log_model(self.rf, "random-forest-class-model")
            mlflow.log_params(self.params)

            # compute evaluation metrics
            acc = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)

            # log metrics
            mlflow.log_metric("accuracy_score", acc)
            mlflow.log_metric("precision", precision)

            # get current run and experiment id
            runID = run.info.run_uuid
            experimentID = run.info.experiment_id

            # create confusion matrix images
            (plt, fig, ax) = Utils.plot_confusion_matrix(y_test,y_pred,y, title="Bank Note Classification Confusion Matrix")

            # create temporary artifact file name and log artifact
            temp_file_name = Utils.get_temporary_directory_path("confusion_matrix-", ".png")
            temp_name = temp_file_name.name
            try:
                fig.savefig(temp_name)
                mlflow.log_artifact(temp_name, "confusion_matrix_plots")
            finally:
                temp_file_name.close()  # Delete the temp file

            # print some data
            print("-" * 100)
            print("Inside MLflow Run with run_id {} and experiment_id {}".format(runID, experimentID))
            print("Estimators trees:", self.params["n_estimators"])
            print(confusion_matrix(y_test,y_pred))
            print(classification_report(y_test,y_pred))
            print("Accuracy Score:", acc)
            print("Precision     :", precision)

            return (experimentID, runID)

#
# Lab/Homework for Some Experimental runs
#
    # 1. Consult RandomForestClassifier documentation
    # 2. Change or add parameters, such as depth of the tree or random_state: 42 etc.
    # 3. Change or alter the range of runs and increments of n_estimators
    # 4. Check in MLfow UI if the metrics are affected
    # 5. Log confusion matirx, recall and F1-score as metrics
    # Nice blog: https://joshlawman.com/metrics-classification-report-breakdown-precision-recall-f1/

if __name__ == '__main__':
    # load and print dataset
    dataset = Utils.load_data("data/bill_authentication.csv")
    Utils.print_pandas_dataset(dataset)
    # iterate over several runs with different parameters
    # TODO in the Lab (change these parameters, n_estimators and random_state
    # with each iteration.
    # Does that change the metrics and accuracy?
    # start with n=10, step by 10 up to X <=100
    for n in range(10, 30, 10):
        params = {"n_estimators": n, "random_state": 0 }
        rfr = RFCModel.new_instance(params)
        (experimentID, runID) = rfr.mlflow_run(dataset)
        print("MLflow Run completed with run_id {} and experiment_id {}".format(runID, experimentID))
        print("-" * 100)

