import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
import idx2numpy as id2n
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def read_file(path):
    return id2n.convert_from_file(path)


def return_prediction(prediction):
    return np.where(prediction == np.max(prediction))[0]


tf.random.set_seed(1)

x_train = read_file('./train/train-images.idx3-ubyte')
y_train = read_file('./train/train-labels.idx1-ubyte')
x_test = read_file('./test/t10k-images.idx3-ubyte')
y_test = read_file('./test/t10k-labels.idx1-ubyte')

x_train = x_train.reshape(60000, 784).astype('float32') / 255
y_train = y_train.astype('float32')

x_test = x_test.reshape(10000, 784).astype('float32')/255
y_test = y_test.astype('float32')

inputs = keras.Input(shape=np.shape(x_train)[1])
layer_1 = layers.Dense(64, activation="relu")(inputs)
layer_2 = layers.Dense(32, activation="relu")(layer_1)
outputs = layers.Dense(10, activation="softmax")(layer_2)

model = keras.Model(inputs=inputs, outputs=outputs)

model.compile(
    optimizer=keras.optimizers.Adam(),
    loss=keras.losses.SparseCategoricalCrossentropy(),
    metrics=[keras.metrics.SparseCategoricalAccuracy()]
)

history = model.fit(
    x_train,
    y_train,
    batch_size=64,
    epochs=5,
    validation_split=0.2
)

print('model validation')

model.evaluate(x=x_test, y=y_test)

plt.imshow(x_test[10].reshape(28, 28) * 255)
plt.show()

print('NN bets it is ', return_prediction(model.predict(np.swapaxes(np.expand_dims(x_test[10], axis=1), 0, 1))))




