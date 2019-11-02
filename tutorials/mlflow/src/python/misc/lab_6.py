import mlflow
import mlflow.sklearn
import mlflow.pyfunc

from lab_utils import Utils
from sklearn.preprocessing import StandardScaler


class MLflowOps():
    def __init__(self):
        #
        # dictionary for mapping model types to their respective load_model calls
        #
        self._model_funcs = {
            'sklearn':mlflow.sklearn.load_model,
            'pyfunc': mlflow.pyfunc.load_model}
        return

    def get_model(self, model_type):
        """
        Method to return the respective model function to load
        :param model_type: string, type of model For example, "sklearn", "keras", "pyfunc" etc
        :return: load_model function to the model type
        """
        return self._model_funcs[model_type]

if __name__ == '__main__':
    mclnt = MLflowOps()
    dataset = Utils.load_data("data/test_petrol_consumption.csv")
    # get all rows and columns but the last column
    X_test = dataset.iloc[:, 0:4].values
    # get all the last columns, which is what we want to predict
    y_test = dataset.iloc[:, 4].values
    print("Observed values {}".format(y_test))
    sc = StandardScaler()
    X_test = sc.fit_transform(X_test)
    # Insert your run_id here in the list
    for run_id in ['c3acc1f495d74bae8c8d5f02766d9dd7']:
        uri = "runs:/" + run_id + "/random-forest-reg-model"
        sk_model = mclnt.get_model("sklearn")(uri)
        print("-" * 100)
        print("Using Scikit-Learn Model Prediction:{}".format(type(sk_model)))
        y_pred = sk_model.predict(X_test)
        print(y_pred)
        py_model = mclnt.get_model("pyfunc")(uri)
        print("Using Pyfunc Model Prediction:{}".format(type(py_model)))
        y_pred = py_model.predict(X_test)
        print(y_pred)
        print("-" * 100)

