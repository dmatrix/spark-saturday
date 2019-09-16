import mlflow
"""
Read documentation on https://mlflow.org/docs/latest/python_api/mlflow.html#mlflow.run
"""
if __name__ == '__main__':
    params = {'alpha': 0.5,
              'l1_ratio': 0.01}
    mlflow.run("git://github.com/mlflow/mlflow-example.git", use_conda=False, parameters=params)
