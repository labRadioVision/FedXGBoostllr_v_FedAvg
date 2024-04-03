import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import numpy as np
import random
import math

# np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)
# from Classes.Params import param
# from Classes.Params import simul_param, fl_param

import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.impute import SimpleImputer
#from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight


#tf.random.set_seed(param.SEED)
#np.random.seed(param.SEED)

global IID
IID = True

def stroke_preprocess(df):
    df.drop(['id'], axis=1, inplace=True)

    cat_features = ['gender', 'hypertension', 'heart_disease', 'ever_married', 'work_type', 'Residence_type',
                    'smoking_status']
    cat_df = df[cat_features]
    cat_dummy = pd.get_dummies(cat_df)

    num_features = ['age', 'avg_glucose_level', 'bmi']
    num_df = df[num_features]
    imp_median = SimpleImputer(missing_values=np.nan, strategy='median')
    num_df = pd.DataFrame(imp_median.fit_transform(num_df), columns=num_df.columns)
    num_scaled = pd.DataFrame(MinMaxScaler().fit_transform(num_df), columns=num_df.columns)

    df = pd.concat([num_scaled, cat_dummy, df['stroke']], axis=1).reset_index(drop=True)

    #y = df['stroke']
    #x = df.drop(['stroke'], axis=1)

    return df #x, y


class Stroke_kaggle:

    def __init__(self, device_index, start_samples, samples, random_data_distribution=1):#0

        df1 = pd.read_csv('Classes/Datasets/healthcare-dataset-stroke-data_new.csv')
        df = stroke_preprocess(df1)

        self.samples_train = samples  # training samples per device
        self.start_samples = start_samples
        self.samples_valid = int(0.4 * samples)  # testing samples per device

        # NEW
        # same dataset for every device, since seed is fixed
        df_valid = df.sample(self.samples_valid, random_state=1)  # sample randomly

        y_test = df_valid['stroke']  # make it a test set
        x_test = df_valid.drop(['stroke'], axis=1)

        df = df.drop(df_valid.index)  # remove them from the set

        y = df['stroke']
        x = df.drop(['stroke'], axis=1)

        x_train, _, y_train, _ = train_test_split(x, y, test_size=0.25, stratify = y, shuffle=True)

        self.num_features = x_train.shape[1]

        self.device_index = device_index


        self.validation_train = x_train.shape[0]# size of the training set
        self.validation_test = x_test.shape[0]# size of the testing set


        # get 'self.start_samples' for training
        if random_data_distribution == 1:
            s_list_train = random.sample(range(self.validation_train), self.samples_train)
        else:
            s_list_train = np.arange(self.start_samples, self.samples_train + self.start_samples)

        s_list_valid = random.sample(range(self.validation_test), self.samples_valid)

        # self.samples for training
        self.x_train = x_train.iloc[s_list_train]  # DATA PARTITION
        self.y_train = y_train.iloc[s_list_train]
        #print('self.x_train.shape', self.x_train.shape)
        #print('self.y_train.shape', self.y_train.shape)

        self.x_test = x_test.iloc[s_list_valid]
        self.y_test = y_test.iloc[s_list_valid]

        #print('self.x_test.shape', self.x_test.shape)
        #print('self.x_test.columns', self.x_test.columns)
        #print('self.y_test.shape', self.y_test.shape)


        print('Number training samples for device {}: '.format(self.device_index) + str(self.samples_train))
        print('Number testing samples for device {}: '.format(self.device_index) + str(self.samples_valid))

        print('TRAIN: class 1: {}; class 0: {}'.format(self.y_train[self.y_train == 1].shape[0], self.y_train[self.y_train == 0].shape[0]))
        print('TEST: class 1: {}; class 0: {}'.format(self.y_test[self.y_test == 1].shape[0], self.y_test[self.y_test == 0].shape[0]))

        del x_test, x_train, y_test, y_train, x, y, df

    def getTrainingData(self, batch_size):
        s_list = random.sample(range(self.samples_train), batch_size)
        batch_xs = self.x_train.iloc[s_list,:]
        batch_ys = self.y_train.iloc[s_list]
        return batch_xs, batch_ys

    def getRandomTestData(self, batch_size):
        s_list = random.sample(range(self.samples_valid), batch_size)

        batch_xs = self.x_test.iloc[s_list]
        batch_ys = self.y_test.iloc[s_list]

        return batch_xs, batch_ys

    def getTestData(self, batch_size, batch_number):
        s_list = np.arange(batch_number * batch_size, (batch_number + 1) * batch_size)
        #print(s_list)
        batch_xs = self.x_test.iloc[s_list]  # self.x_train[s_list, :, :, 0]
        batch_ys = self.y_test.iloc[s_list]
        #batch_xs = self.x_test[s_list, :, :, 0]
        #batch_ys = self.y_test[s_list]
        return batch_xs, batch_ys

    def preprocess_observation(self, obs, batch_size):
        #img = obs  # crop and downsize
        #img = (img).astype(np.float)
        #print(obs.shape)
        return obs#img.reshape(batch_size, 28, 28, 1)

    def return_input_shape(self):
        return self.num_features

    def get_class_weights(self):
        # GENERIC VERSION
        #class_weights = compute_class_weight(class_weight ='balanced', classes = np.unique(self.y_train), y =self.y_train)
        #return dict(zip(np.unique(self.y_train), class_weights))

        mu = 0.15
        labels_dict = {i :  self.y_train[self.y_train == i].shape[0] for i in np.unique(self.y_train)}

        total = np.sum(list(labels_dict.values()))
        keys = labels_dict.keys()
        class_weight = dict()

        for key in keys:
            score = math.log(mu * total / float(labels_dict[key]))
            class_weight[key] = score if score > 1.0 else 1.0

        return class_weight
