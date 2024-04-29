import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
import tensorflow as tf
import numpy as np
from classes.params import fl_param, simul_param

class Dataset:
    def __init__(self, device_index):
        # TODO: pass the folder name as a parameter
        
        self.x_train_local = np.load(f'data/client_{device_index}/train/x_train.npy', allow_pickle=True)
        self.y_train_local = np.load(f'data/client_{device_index}/train/y_train.npy')
        #print(self.y_train_local)
        self.x_valid = np.load(f'data/client_{device_index}/valid/x_valid.npy', allow_pickle=True)
        self.y_valid = np.load(f'data/client_{device_index}/valid/y_valid.npy')
        
        #samp_per_class = np.sum(self.y_train_local, axis=0).astype(int)
        #self.num_samples = np.sum(samp_per_class).astype(int)
        self.num_samples = len(self.x_train_local)
        self.batch_size = fl_param.BATCH_SIZE if fl_param.BATCH_SIZE < len(self.x_train_local) else len(self.x_train_local)
        

    def _info(self):
        print('Samples per class:', np.sum(self.y_train_local, axis=0).astype(int))
        print('Total samples:', self.num_samples)

        
    def get_train_dataset(self, num_batches):
        train_dataset = tf.data.Dataset.from_tensor_slices((self.x_train_local, self.y_train_local))
        train_dataset = train_dataset.shuffle(buffer_size=len(self.x_train_local)) 
        train_dataset = train_dataset.shuffle(len(self.x_train_local)).batch(self.batch_size)
        train_dataset = train_dataset.take(num_batches)
        return train_dataset

    def get_test_dataset(self):
        test_dataset = tf.data.Dataset.from_tensor_slices((self.x_valid, self.y_valid))
        test_dataset = test_dataset.shuffle(buffer_size=len(self.x_valid)) 
        test_dataset = test_dataset.batch(self.batch_size)
        return test_dataset

    def return_input_output(self):
        return self.x_train_local.shape[1:], np.unique(self.y_valid, axis=0).shape[0] if np.unique(self.y_valid, axis=0).shape[0] > 2 else 1