import mlflow
import mlflow.sklearn
import mlflow.pyfunc
from sklearn.preprocessing import StandardScaler
from lab_utils import Utils

class MLflowOps():
    """
    Class for some operations on model like loading, reproducing or predicting results
    This class could be extend to implement other operations:
    1. Execute or run methods from  MLflow Command LIne interface using Click Python module
    2. Execute any methods using click.CLIRunner()
    3. Deploy or serve model usng MLflow CLI interface
    """
    def __init__(self):
        #
        # dictionary for mapping model types to their respective load_model calls
        # TODO in Lab
        # Extend the dictionary mapping to include mlflow.pyfunc.load_model
        #
        self._model_funcs = {
            'sklearn':mlflow.sklearn.load_model}

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
    # get all values only from the last columns, which is what we want to predict
    y_test = dataset.iloc[:, 4].values
    print("Observed values {}".format(y_test))
    sc = StandardScaler()
    X_test = sc.fit_transform(X_test)
    print("-" * 100)
    #
    # TODO in Lab
    # Add your run_uids from Lab-1 Runs. <cut-and-past in the list>
    # Can you try Lab-1 runs with model `random-forest-class-mode`, our classification model
    # and use its respective run_uids and test data
    for run_id in ['6238ccf85a8e418d8017ff26fdae9452']:
        uri = "runs:/" + run_id + "/random-forest-reg-model"
        sk_model = mclnt.get_model("sklearn")(uri)
        print("Using Sckit-Learn Model Prediction:{}".format(type(sk_model)))
        print("Run_uui={}".format(run_id))
        y_pred = sk_model.predict(X_test)
        print(y_pred)
        #
        # TODO in Lab
        # Add code as above to load model as PyFunc
        # Use the same test data as above to predict
        #
    print("-" * 100)
