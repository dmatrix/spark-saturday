import argparse

import keras
import tensorflow as tf
from keras import models
from keras import layers

import mlflow
import mlflow.keras
from mlflow.tracking import MlflowClient

"""
Aim of this module is:
1. Introduce tracking ML experiments in MLflow
2. Create your own experiment name and log runs under it
3. Record parameters, metrics, and a model
"""

# TODO in LAB
#    Add layers
#    Make hidden units larger
#    Try a different optimizer: RMSprop and Adadelta
#    Train for more epochs
#    change these default parameters are and observe how it will effect the results
#    or provide them on command line.
#    For exaxmple,
#    `python keras_mnist_lab_5.py --epochs 10`
#    `python keras_mnist_lab_5.py -e 10 -n 128`
#
parser = argparse.ArgumentParser(
    description='Train a Keras feed-forward network for MNIST classification in TensorFlow/Keras')
parser.add_argument('--batch_size', '-b', type=int, default=128)
parser.add_argument('--epochs', '-e', type=int, default=5)
parser.add_argument('--learning_rate', '-l', type=float, default=0.05)
parser.add_argument('--num_hidden_units', '-n', type=int, default=512)
parser.add_argument('--num_hidden_layers', '-N', type=int, default=1)
parser.add_argument('--dropout', '-d', type=float, default=0.25)
parser.add_argument('--momentum', '-m', type=float, default=0.85)
args = parser.parse_args()

# get MNIST data set
mnist = keras.datasets.mnist
# normalize the dataset
(x_train, y_train),(x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# get an experiment name to see if it exists before creating one
# MLflow creates run under the name "Default," with experiment id 0
# Let's create our own experiment ID and name
# Get handle to MLflowClient
client = MlflowClient()
exp_id = 0
entity = client.get_experiment_by_name("MLflow Strata Tutorial")
if entity is None:
    exp_id = mlflow.create_experiment("MLflow Strata Tutorial")
else:
    exp_id = entity.experiment_id

def mlfow_run(run_name="Lab-5:Keras_MNIST", experiment_id=exp_id, model_summary=False):
    """
    Method to run MLflow experiment
    :return: Tuple (experiment_id, run_id)
    """
    with mlflow.start_run(run_name=run_name,  experiment_id = exp_id) as run:

        model = models.Sequential()
        #
        # The first layer in this network transforms the format of the images from a 2d-array (of 28 by 28 pixels),
        # to a 1d-array of 28 * 28 = 784 pixels.
        model.add(layers.Flatten(input_shape=x_train[0].shape))
        # add extra hidden layers to expand the NN
        # --num_hidden_layers or -N  in the command line arguments
        for n in range(0, args.num_hidden_layers):
            model.add(layers.Dense(args.num_hidden_units, activation=tf.nn.relu))
        # dropout is an regularization technique for NN where we randomly dropout a layer if the
        # computed gradients are minimal or have no effect.
        model.add(layers.Dropout(args.dropout))
        # final layer with softmax activation layer
        model.add(layers.Dense(10, activation=tf.nn.softmax))
        if model_summary:
            model.summary()
        # Use Scholastic Gradient Descent (SGD)
        # https://keras.io/optimizers/
        #
        optimizer = keras.optimizers.SGD(lr=args.learning_rate,
                                         momentum=args.momentum,
                                         nesterov=True)

        # compile the model with optimizer and loss type
        # common loss types for classification are
        # 1. sparse_categorical_crossentropy
        # 2. binary_crossentropy
        model.compile(optimizer=optimizer,
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])

        # Callback function for logging metrics at at of each epoch
        class LogMetricsCallback(keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs={}):
                mlflow.log_metric("training_loss", logs["loss"], epoch)
                mlflow.log_metric("training_accuracy", logs["acc"], epoch)

        # fit the model
        # get experiment id and run id
        runID = run.info.run_uuid
        experimentID = run.info.experiment_id
        print("-" * 100)
        print("MLflow Run with run_id {} and experiment_id {}".format(runID, experimentID))
        model.fit(x_train, y_train,
                  epochs=args.epochs,
                  batch_size=args.batch_size,
                  callbacks=[LogMetricsCallback()])
        # evaluate the model
        test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
        # log metrics
        mlflow.log_metric("test_loss", test_loss)
        mlflow.log_metric("test_accuracy", test_acc)

        # <TODO in LAB>
        # log parameters used in the command line args use in the model
        #

        # log model as native Keras Model
        mlflow.keras.log_model(model, artifact_path="keras-model")

        # get experiment id and run id
        runID = run.info.run_uuid
        experimentID = run.info.experiment_id

        return (experimentID, runID)

if __name__ == '__main__':
    (experimentID, runID) = mlfow_run(run_name="Jules-Lab5:Keras_MNIST")
    print("MLflow completed with run_id {} and experiment_id {}".format(runID, experimentID))
    print(tf.__version__)
    print(mlflow.__version__)
    print("-" * 100)

