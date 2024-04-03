# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import tensorflow as tf
from classes.params import simul_param


class VanillaMLP:

    def __init__(self, input_shape, output_shape):

        self.input_shape = input_shape
        self.output_shape = output_shape
    # Model definition
    def return_model(self):

        model = tf.keras.Sequential()

        model.add(tf.keras.layers.Dense(32, activation='relu', input_shape=self.input_shape))#
        model.add(tf.keras.layers.Dense(16, activation='relu'))
        model.add(tf.keras.layers.Dense(self.output_shape, activation='sigmoid'))

        return model