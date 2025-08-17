import tensorflow as tf
from tensorflow.keras import layers

dummy_model = tf.keras.Sequential([
    layers.Input(shape=(900,12)),
    layers.Dense(3, activation='softmax')
])
dummy_model.save("models/49_model.keras")

