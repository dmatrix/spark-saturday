'''

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

Aim of this Lab:

1. Understand MLflow Tracking API
2. How to use the MLflow Tracking API
3. Use the MLflow API to experiment several Runs
4. Interpret and observer runs via the MLflow UI

https://stackabuse.com/random-forest-algorithm-with-python-and-scikit-learn/
'''

import mlflow.sklearn
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score
from lab_utils import load_data, plot_graphs, get_mlflow_directory_path, print_pandas_dataset, plot_confusion_matrix

class RFCModel():

    # class wide variables common to all instances

    def __init__(self, params={}):
        self.rf = RandomForestClassifier(**params)
        self.params = params

    def model(self):
        return self.rf

    def mlflow_run(self, df, r_name="RF Bank Note Classification Experiment"):

        with mlflow.start_run(run_name=r_name) as run:
            # get all rows and columns but the last column
            X = df.iloc[:, 0:4].values
            # get all the last columns, which is what we want to predict
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
            (plt, fig, ax) = plot_confusion_matrix(y_test,y_pred,y, title="Bank Note Classification Confusion Matrix")
            image_dir = get_mlflow_directory_path(experimentID, runID, "images")
            save_image = os.path.join(image_dir, "confusion_matrix.png")
            fig.savefig(save_image)

            # log artifact
            mlflow.log_artifacts(image_dir, "images")

            # print some data
            print("Inside MLflow Run with run_id {} and experiment_id {}".format(runID, experimentID))
            print("Estimators trees:", self.params["n_estimators"])
            print(confusion_matrix(y_test,y_pred))
            print(classification_report(y_test,y_pred))
            print("Accuracy Score:", acc)
            print("Precision     :", precision)
            print("-" * 100)

# Lab/Homework for Some Experimental runs
    # 1. Consult RandomForestClassifier documentation
    # 2. Change or add parameters, such as depth of the tree or random_state: 42 etc.
    # 3. Change or alter the range of runs and increments of n_estimators
    # 4. Check in MLfow UI if the metrics are affected
    # 5. Log confusion matirx, recall and F1-score as metrics
    # Nice blog: https://joshlawman.com/metrics-classification-report-breakdown-precision-recall-f1/

if __name__ == '__main__':
    # load and print dataset
    dataset = load_data("data/bill_authentication.csv")
    print_pandas_dataset(dataset)
    # iterate over several runs with different parameters
    for n in range(25, 350, 25):
        params = {"n_estimators": n, "random_state": 0 }
        rfr = RFCModel(params)
        rfr.mlflow_run(dataset)
