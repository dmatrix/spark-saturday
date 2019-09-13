# in case this is run outside of conda environment with python2
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import mlflow
from mlflow import pyfunc
import pandas as pd
import shutil
import tempfile
import tensorflow as tf
from tensorflow.python.saved_model import tag_constants
import mlflow.tensorflow

# Enable auto-logging to MLflow to capture TensorBoard metrics.
mlflow.tensorflow.autolog()

def main(argv):
    # Builds, trains and evaluates a tf.estimator. Then, exports it for inference, logs the exported model
    # with MLflow, and loads the fitted model back as a PyFunc to make predictions.
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.boston_housing.load_data()
    # There are 13 features we are using for inference.
    feat_cols = [tf.feature_column.numeric_column(key="features", shape=(x_train.shape[1],))]
    feat_spec = {
        "features": tf.placeholder("float", name="features", shape=[None, x_train.shape[1]])}
    hidden_units = [1024,256,64,16]
    steps = 1000
    nepochs = [10, 15, 20, 25]
    for run_n in [0, 1, 2, 3]:
        regressor = tf.estimator.DNNRegressor(hidden_units=hidden_units, feature_columns=feat_cols)
        train_input_fn = tf.estimator.inputs.numpy_input_fn({"features": x_train}, y_train,
                                                         num_epochs=nepochs[run_n], shuffle=True)
        with mlflow.start_run() as run :
            runID = run.info.run_uuid
            experimentID = run.info.experiment_id

            # log parameters
            mlflow.log_param("Hidden Units", hidden_units)
            mlflow.log_param("Steps", steps)
            mlflow.log_param("epochs", nepochs[run_n])
            regressor.train(train_input_fn, steps=steps)
            test_input_fn = tf.estimator.inputs.numpy_input_fn({"features": x_test}, y_test,
                                                       num_epochs=nepochs[run_n], shuffle=True)
            # Compute mean squared error
            mse = regressor.evaluate(test_input_fn, steps=steps)
            mlflow.log_metric("Average Mean Square Error", mse['average_loss'])
            mlflow.log_metric("Mean Square Error", mse['loss'])
            # Building a receiver function for exporting
            receiver_fn = tf.estimator.export.build_raw_serving_input_receiver_fn(feat_spec)
            temp = tempfile.mkdtemp()
            try:
                saved_estimator_path = regressor.export_savedmodel(temp, receiver_fn).decode("utf-8")
                # Logging the saved model
                mlflow.tensorflow.log_model(tf_saved_model_dir=saved_estimator_path,
                                            tf_meta_graph_tags=[tag_constants.SERVING],
                                            tf_signature_def_key="predict",
                                            artifact_path="model")
                # Reloading the model
                pyfunc_model = pyfunc.load_pyfunc(mlflow.get_artifact_uri('model'))
                df = pd.DataFrame(data=x_test, columns=["features"] * x_train.shape[1])
                # Predicting on the loaded Python Function
                predict_df = pyfunc_model.predict(df)
                predict_df['original_labels'] = y_test

                # print some data
                print("Inside MLflow Run with run_id {} and experiment_id {}".format(runID, experimentID))
                print('Mean Squared Error               :',  mse['average_loss'])
                print('Mean Squared Error/per batch     :',  mse['loss'])
                print('Evaluate MSE Dictionary          :', mse)

                print("-" * 100)
                print(predict_df)
            finally:
                shutil.rmtree(temp)


if __name__ == "__main__":
    # The Estimator periodically generates "INFO" logs; make these logs visible.
    tf.logging.set_verbosity(tf.logging.INFO)
    # Enable auto-logging to MLflow to capture TensorBoard metrics.
    mlflow.tensorflow.autolog()
    tf.app.run(main=main)
