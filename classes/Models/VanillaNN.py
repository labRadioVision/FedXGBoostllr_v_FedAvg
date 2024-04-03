# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import tensorflow as tf

# MacOs
# import os
# import plaidml.keras
# plaidml.keras.install_backend()
# os.environ['KERAS_BACKEND'] = 'plaidml.keras.backend'
# import keras

from classes.params import simul_param


class VanillaNN:

    def __init__(self, input_shape):

        self.input_shape = input_shape

    # Model definition
    def return_model(self):

        model = tf.keras.Sequential()

        model.add(tf.keras.layers.Dense(32, activation='relu', input_shape=(self.input_shape,)))#
        model.add(tf.keras.layers.Dense(16, activation='relu'))
        model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

        return model