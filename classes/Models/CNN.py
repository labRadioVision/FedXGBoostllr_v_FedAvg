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


class CNN:
    
    def __init__(self, input_shape, output_shape):
        print('CNN model')
        self.input_shape = input_shape
        self.output_shape = output_shape

    # Model definition
    def return_model(self):

        start_f = 8
        depth = 2 # 5
        model = tf.keras.Sequential()

        # Features extraction
        for i in range(depth):
            if i == 0:
                input_shape = self.input_shape
            else:
                input_shape=[None]

            # Conv block: Conv2D -> Activation -> Pooling
            model.add(tf.keras.layers.Conv2D(filters=start_f, 
                                            kernel_size=(3, 3),
                                            strides=(1, 1),
                                            padding='same',
                                            input_shape=input_shape))
            #model.add(tf.keras.layers.ReLU())
            model.add(tf.keras.layers.LeakyReLU())
            model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2)))

            start_f *= 2

        # Classifier
        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dense(units=16))  # 512
        model.add(tf.keras.layers.LeakyReLU())
        model.add(tf.keras.layers.Dense(units=self.output_shape, activation='sigmoid'))
        return model
