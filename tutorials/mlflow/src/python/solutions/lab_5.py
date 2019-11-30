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

# Set experiment name; if one does exist, one will be created for you
# Use MlflowClient() to fetch the experiment details

client = MlflowClient()

mlflow.set_experiment("MLflow ODSC Tutorial")
entity = client.get_experiment_by_name("MLflow ODSC Tutorial")
exp_id = entity.experiment_id

def get_optimizer(opt_name):
    """
    :param name: name of the Keras optimizer
    :param args: args for the optimizer
    :return: Keras optimizer
    """
    if opt_name == 'SGD':
        optimizer = keras.optimizers.SGD(lr=args.learning_rate,
                                         momentum=args.momentum,
                                         nesterov=True)
    elif (opt_name == 'RMSprop'):
        optimizer = keras.optimizers.RMSprop(lr=args.learning_rate, rho=0.9, epsilon=None, decay=0.0)
    else:
        optimizer = keras.optimizers.Adadelta(lr=args.learning_rate, epsilon=None, decay=0.0)

    return optimizer

def mlfow_run(run_name="Lab-5:Keras_MNIST", model_summary=False, opt_name="SGD"):
    """
    :param run_name: name of the run
    :param experiment_id: experiment id under which to create this run
    :param model_summary: print model summary; default is False
    :param opt_name: name of the optimizer to use
    :return: Keras optimizer
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
        optimizer = get_optimizer(opt_name)

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
        [mlflow.log_param(arg, getattr(args, arg)) for arg in vars(args)]
        mlflow.log_param("optimizer", opt_name)
        mlflow.log_param("loss_function", "sparse_categorical_crossentropy")


        # log model as native Keras Model
        mlflow.keras.log_model(model, artifact_path="keras-model")

        # get experiment id and run id
        runID = run.info.run_uuid
        experimentID = run.info.experiment_id

        return (experimentID, runID)

if __name__ == '__main__':
    for opt_name in ['SGD', 'RMSprop','Adadelta']:
        (experimentID, runID) = mlfow_run(run_name="Jules-Lab5:Keras_MNIST", opt_name=opt_name)
        print("MLflow completed with run_id {}, optimizer {} and experiment_id {}".format(runID, opt_name, experimentID))
        print("-" * 100)
    print(tf.__version__)
    print(mlflow.__version__)

