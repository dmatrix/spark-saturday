import mlflow
import mlflow.sklearn
import mlflow.pyfunc
import pandas as pd

from lab_utils import load_data

class MLflowOperations():
    def __init__(self):
        #
        # dictionary for mapping model types to their respective load_model calls
        #
        self._model_funcs = {
            'sklearn':mlflow.sklearn.load_model,
            'pyfunc': mlflow.pyfunc.load_model}
        return

    def get_model(self, model_type):
        '''
        Method to return the respective model function to load
        :param model_type: string, type of model For example, "sklearn", "keras", "pyfunc" etc
        :return: load_model function to the model type
        '''
        return self._model_funcs[model_type]

if __name__ == '__main__':
    mclnt = MLflowOperations()
    dataset = load_data("data/test_petrol_consumption.csv")
    # get all rows and columns but the last column
    X_test = dataset.iloc[:, 0:4].values
    # get all the last columns, which is what we want to predict
    y_test = dataset.iloc[:, 4].values
    print("Observed values {}".format(y_test))
    for run_id in ['c0c2010c793345dd80d88da5cac6cdcf']:
        uri = "runs:/" + run_id + "/random-forest-reg-model"
        sk_model = mclnt.get_model("sklearn")(uri)
        print("-" * 100)
        print("Using Sckit-Learn Model Prediction:{}".format(type(sk_model)))
        y_pred = sk_model.predict(X_test)
        print(y_pred)
        py_model = mclnt.get_model("pyfunc")(uri)
        print("Using Pyfunc Model Prediction:{}".format(type(py_model)))
        y_pred = py_model.predict(X_test)
        print(y_pred)
        print("-" * 100)

