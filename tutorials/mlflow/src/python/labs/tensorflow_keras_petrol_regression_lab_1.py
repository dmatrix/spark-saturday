import tensorflow as tf
import mlflow
import mlflow.tensorflow

from sklearn.model_selection import train_test_split

from lab_utils import Utils
from sklearn.preprocessing import StandardScaler


class TFKerasRegModel:

    def __init__(self, params={}):
        self._params = params

    @property
    def params(self, params):
        self._params = params

    @params.getter
    def params(self):
        return self._params

    def add_parameter(self, key, value):
        self._params[key] = value

    def update_parameter(self, key, value):
        self.add_parameter(key, value)

    def get_parameter(self, key):
        try:
            return self._params[key]
        except KeyError as e:
            return None

    def model(self):
        model_attr = "tf_keras_reg_model"
        model_reg = getattr(self, model_attr, None)
        if model_reg is None:
            try:
                model_reg = self._build_compiled_model()
                setattr(self, model_attr, model_reg)
            except Exception as e:
                raise Exception(e)
            return model_reg

    def _build_compiled_model(self):
        # build the model
        km = tf.keras.models.Sequential([
                tf.keras.layers.Dense(self.get_parameter('input_units'),
                    input_shape=self.get_parameter('input_shape'),
                    activation=self.get_parameter('activation'),
                    name="input_layer"),
                 tf.keras.layers.Dense(self.get_parameter('input_units'),
                                activation=self.get_parameter('activation'),
                                name="hidden_layer_1"),
                tf.keras.layers.Dense(1)])
        # compile the model
        km.compile(loss=self.get_parameter('loss'),
                   optimizer=self.get_parameter('optimizer'),
                   metrics=['mse', Utils.rmse])
        return km

    def mlflow_run(self, X, y, run_name="TF_Keras_Regression"):
        # create train and test data
        with mlflow.start_run(run_name=run_name)as run:
            # Automatically capture the model's parameters, metrics, artifacts,
            # and source code with the `autolog()` function
            mlflow.tensorflow.autolog()
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
            sc = StandardScaler()
            X_train = sc.fit_transform(X_train)
            X_test = sc.fit_transform(X_test)
            self.model().fit(X_train, y_train,
                            epochs= self.get_parameter('epochs'),
                            batch_size=self.get_parameter('batch_size'),
                            validation_split=.2)
            runID = run.info.run_uuid
            experimentID = run.info.experiment_id

            return (experimentID, runID)

if __name__ =='__main__':
    print("Using TensorFlow Version={}".format(tf.__version__))
    params_list = [
        {'input_units': 64,
              'input_shape': (4,),
              'activation': 'relu',
              'optimizer': 'adam',
              'loss': 'mse',
              'epochs': 100,
              'batch_size': 128},
        {'input_units': 128,
              'input_shape': (4,),
              'activation': 'relu',
              'optimizer': 'adam',
              'loss': 'mse',
              'epochs': 200,
              'batch_size': 128},
        {'input_units': 256,
            'input_shape': (4,),
            'activation': 'relu',
            'optimizer': 'adam',
            'loss': 'mse',
            'epochs': 200,
            'batch_size': 128}
        ]

    dataset = Utils.load_data("data/petrol_consumption.csv")
    # get all feature independent attributes
    X = dataset.iloc[:, 0:4].values
    # get all the values of last columns, dependent variables,
    # which is what we want to predict as our values, the petrol consumption
    y = dataset.iloc[:, 4].values
    for params in params_list:
        keras_model = TFKerasRegModel(params)
        (experimentID, runID) = keras_model.mlflow_run(X, y)
        print("MLflow completed with run_id {} and experiment_id {}".format(runID, experimentID))
