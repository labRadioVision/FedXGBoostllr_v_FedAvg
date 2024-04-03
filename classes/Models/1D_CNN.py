import tensorflow.keras as tfk


from classes.params import fl_param


class CNN:
    
    def __init__(self, input_shape, output_shape):
        print('1D_CNN model')
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.trees_client = 15
    # Model definition
    def return_model(self):

        model = tfk.models.Sequential()
        
        
        model.add(
        tfk.layers.Conv1D(
            filters = 16,
            kernel_size=self.trees_client,
            strides=self.trees_client,
            activation="relu",
            input_shape=(fl_param.NUM_CLIENTS * self.trees_client, 1),
        )
    )
        model.add(tfk.layers.Flatten())
        model.add(tfk.layers.Dense(fl_param.NUM_CLIENTS * self.trees_client, activation="relu"))
        
        model.add(tfk.layers.Dense(self.output_shape, activation="softmax"))
        #model.add(tfk.layers.Dense(1, activation="sigmoid"))
        return model
