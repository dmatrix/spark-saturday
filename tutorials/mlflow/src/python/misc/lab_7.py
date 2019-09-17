import mlflow
"""
Read documentation on https://mlflow.org/docs/latest/python_api/mlflow.html#mlflow.run
"""
if __name__ == '__main__':
    params_1 = {'alpha': 0.5,
              'l1_ratio': 0.01}
    params_2 = {'epochs': 5}

    # Two runs from GitHub Project
    mlflow.run("git://github.com/mlflow/mlflow-example.git", use_conda=False, parameters=params_1)
    mlflow.run("git://github.com/dmatrix/mlflow-example.git", use_conda=False, parameters=params_2)
