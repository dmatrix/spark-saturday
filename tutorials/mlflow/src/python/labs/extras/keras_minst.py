import tensorflow as tf
import mlflow
from mlflow import pyfunc
import pandas as pd
import numpy as np
from mlflow.keras import log_model
mnist = tf.keras.datasets.mnist

print("Using TensorFlow={}".format(tf.__version__))
(x_train, y_train),(x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(512, activation=tf.nn.relu),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=1)
model.evaluate(x_test, y_test)
#log model
log_model(model, "keras-model")
tf_k_model = mlflow.keras.load_model(mlflow.get_artifact_uri('keras-model'))
for i in range(3):
    # get single images in the shape expected by the model
    img = x_test[i]
    img = (np.expand_dims(img,0))
    print(img.shape)
    predictions_single = tf_k_model.predict(img)
    print(predictions_single)
    prediction_result = np.argmax(predictions_single[0])
    print("predicted value={};observed value={}".format(prediction_result, y_test[i]))
"""
pyfunc_model = pyfunc.load_pyfunc(mlflow.get_artifact_uri('keras-model'))
df = pd.DataFrame(data=x_test, columns=["features"] * x_train.shape[1])
print(pyfunc_model.predict(df))
"""
print("Done!")
