import mlflow
from  mlflow.tracking import MlflowClient

if __name__ == '__main__':
    mlflow.set_tracking_uri("databricks")
    cltn = MlflowClient()
    models = cltn.list_registered_models()
    for m in models:
        print(m)
