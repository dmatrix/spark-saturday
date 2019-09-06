import argparse

import keras
import tensorflow as tf

import mlflow
import mlflow.keras
from mlflow.tracking import MlflowClient

'''
Aim of this module is:
1. Introduce tracking ML experiments in MLflow
2. Create your own experiment name and log runs under it
3. Record parameters, metrics, and a model
'''

# TODO in LAB
# Add layers
#    Make hidden units larger
#    Try a different optimizer: RMSprop and Adadelta
#    Train for more epochs
parser = argparse.ArgumentParser(
    description='Train a Keras feed-forward network for MNIST classification in TensorFlow/Keras')
parser.add_argument('--batch-size', '-b', type=int, default=128)
parser.add_argument('--epochs', '-e', type=int, default=5)
parser.add_argument('--learning-rate', '-l', type=float, default=0.05)
parser.add_argument('--num-hidden-units', '-n', type=int, default=512)
parser.add_argument('--num-hidden-layers', '-N', type=int, default=1)
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

# do the MLflow thing
with mlflow.start_run(run_name="Lab5:Keras_MNIST",  experiment_id = exp_id) as run:

    model = keras.models.Sequential([
      keras.layers.Flatten(input_shape=x_train[0].shape),
      keras.layers.Dense(args.num_hidden_units, activation=tf.nn.relu),
      keras.layers.Dropout(args.dropout),
      keras.layers.Dense(10, activation=tf.nn.softmax)
    ])

    # Use Scholastic Gradient Descent (SGD)
    # https://keras.io/optimizers/
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
    # log parameters

    # log model as native Keras Model
    mlflow.keras.log_model(model, artifact_path="keras-model")

    # get experiment id and run id
    runID = run.info.run_uuid
    experimentID = run.info.experiment_id
    print("MLflow Run with run_id {} and experiment_id {}".format(runID, experimentID))

