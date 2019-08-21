import pandas as pd

df = pd.read_csv("data/airbnb-cleaned-mlflow.csv")

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(df.drop(["price"], axis=1), df[["price"]].values.ravel(), random_state=42)

import mlflow.sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

with mlflow.start_run(run_name="Basic RF Experiment") as run:
    # Create model, train it, and create predictions
    rf = RandomForestRegressor()
    rf.fit(X_train, y_train)
    predictions = rf.predict(X_test)

    # Log model
    mlflow.sklearn.log_model(rf, "random-forest-model")

    # Create metrics
    mse = mean_squared_error(y_test, predictions)
    print("  mse: {}".format(mse))

    # Log metrics
    mlflow.log_metric("mse", mse)

    runID = run.info.run_uuid
    experimentID = run.info.experiment_id

    print("Inside MLflow Run with run_id {} and experiment_id {}".format(runID, experimentID))

from mlflow.tracking import MlflowClient

artifactURL = MlflowClient().get_experiment(experimentID).artifact_location
modelURL = artifactURL + "/" + runID + "/artifacts/random-forest-model"

def log_rf(experimentID, run_name, params, X_train, X_test, y_train, y_test):
    import os
    import matplotlib.pyplot as plt
    import mlflow.sklearn
    import seaborn as sns
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    import tempfile

    with mlflow.start_run(experiment_id=experimentID, run_name=run_name) as run:
        # Create model, train it, and create predictions
        rf = RandomForestRegressor(**params)
        rf.fit(X_train, y_train)
        predictions = rf.predict(X_test)

        # Log model
        mlflow.sklearn.log_model(rf, "random-forest-model")

        # Log params
        [mlflow.log_param(param, value) for param, value in params.items()]

        # Create metrics
        mse = mean_squared_error(y_test, predictions)
        mae = mean_absolute_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)
        print("  mse: {}".format(mse))
        print("  mae: {}".format(mae))
        print("  R2 : {}".format(r2))

        # Log metrics
        mlflow.log_metric("mse", mse)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("r2", r2)

        # Create feature importance
        importance = pd.DataFrame(list(zip(df.columns, rf.feature_importances_)),
                                  columns=["Feature", "Importance"]
                                  ).sort_values("Importance", ascending=False)

        # Log importances using a temporary file
        temp = tempfile.NamedTemporaryFile(prefix="feature-importance-", suffix=".csv")
        temp_name = temp.name
        try:
            importance.to_csv(temp_name, index=False)
            mlflow.log_artifact(temp_name, "feature-importance.csv")
        finally:
            temp.close() # Delete the temp file

        # Create plot
        fig, ax = plt.subplots()

        sns.residplot(predictions, y_test, lowess=True)
        plt.xlabel("Predicted values for Price ($)")
        plt.ylabel("Residual")
        plt.title("Residual Plot")

        # Log residuals using a temporary file
        temp = tempfile.NamedTemporaryFile(prefix="residuals-", suffix=".png")
        temp_name = temp.name
        try:
            fig.savefig(temp_name)
            mlflow.log_artifact(temp_name, "residuals.png")
        finally:
            temp.close() # Delete the temp file

        #display(fig)
        return run.info.run_uuid

params = {
        "n_estimators": 100,
        "max_depth": 5,
        "random_state": 42
}

log_rf(experimentID, "Second Run", params, X_train, X_test, y_train, y_test)

params_1000_trees = {
    "n_estimators": 1000,
    "max_depth": 10,
    "random_state": 42
}

log_rf(experimentID, "Third Run", params_1000_trees, X_train, X_test, y_train, y_test)

from  mlflow.tracking import MlflowClient

client = MlflowClient()

client.list_run_infos(experimentID)
