
from keras import models
from keras import layers
from lab_utils import KIMDB_Data_Utils

import mlflow
import mlflow.keras
from mlflow.tracking import MlflowClient

class KerasBaseModel():

    def __init__(self):
        return

    def model(self, params={}):
        """
        Build the base line model with one input layer, one hidden layer, and one output layer, with
        16, 16, and 1 output neurons. Default activation functions for input and hidden layer are relu
        and sigmoid respectively
        :return: a Keras network model
        """
        base_model = models.Sequential()
        base_model.add(layers.Dense(16, activation='relu', input_shape=(10000,)))
        base_model.add(layers.Dense(16, activation='relu'))
        base_model.add(layers.Dense(1, activation='sigmoid'))
        return base_model

class KerasExperimentModel(KerasBaseModel):
    def __int__(self):
        super().__init__()
        return

    def model(self, params={'hidden_layers':2, 'output':16, 'activation':'relu'}):
        hidden_layers= params['hidden_layers']
        output = params ['output']
        activation=params['activation']
        exp_model = models.Sequential()
        # add the input layers
        exp_model.add(layers.Dense(output, activation=activation, input_shape=(10000,)))
        # add hidden layers
        for i in range(0, hidden_layers):
            exp_model.add(layers.Dense(output, activation=activation))
            # add output layer
        exp_model.add(layers.Dense(1, activation='sigmoid'))

        return exp_model

if __name__ == '__main__':
    # base model
    KerasBaseModel().model().summary()
    # build an experimental model
    KerasExperimentModel().model(params={'hidden_layers':3, 'output':32, 'activation':'sigmoid'}).summary()
    # default without params
    KerasExperimentModel().model().summary()
    # load the imdb data set
    kdata_cls = KIMDB_Data_Utils()
    (train_data, train_labels), (test_data, test_labels) = kdata_cls.fetch_imdb_data(num_words=10000)
    x_train = kdata_cls.prepare_vectorized_sequences(train_data)
    x_test = kdata_cls.prepare_vectorized_sequences(test_data)
    y_train = kdata_cls.prepare_vectorized_labels(train_labels)
    y_test = kdata_cls.prepare_vectorized_labels(test_labels)

