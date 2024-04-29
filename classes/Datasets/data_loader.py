import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
import tensorflow as tf
import  tensorflow.keras.datasets as tfds
import numpy as np
import pandas as pd

import medmnist
import numpy as np
from medmnist import INFO
import itertools
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split

def load_mnist():
    # load the dataset
    (x_train, y_train), (x_test, y_test) = tfds.mnist.load_data(path='mnist.npz')

    x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], 1)
    x_train = x_train / 255.0
    x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 1)
    x_test = x_test / 255.0

    y_train = tf.one_hot(y_train.astype(np.int32), depth=10)
    y_test = tf.one_hot(y_test.astype(np.int32), depth=10)
    
    return x_train, y_train, x_test, y_test

def load_stroke():
    # see preprocessing in stroke_data_prepr.ipynb
    path = 'classes/Datasets/healthcare-dataset-stroke-prep.csv'
    df = pd.read_csv(path)
    #df.drop(['id'], axis=1, inplace=True)

    cat_features = ['gender', 'hypertension', 'heart_disease', 'ever_married', 'work_type', 'Residence_type',
                    'smoking_status']
    cat_df = df[cat_features]
    cat_dummy = pd.get_dummies(cat_df)

    num_features = ['age', 'avg_glucose_level', 'bmi']
    num_df = df[num_features]
    imp_median = SimpleImputer(missing_values=np.nan, strategy='median')
    num_df = pd.DataFrame(imp_median.fit_transform(num_df), columns=num_df.columns)
    num_scaled = pd.DataFrame(MinMaxScaler().fit_transform(num_df), columns=num_df.columns)

    df_clean = pd.concat([num_scaled, cat_dummy, df['stroke']], axis=1).reset_index(drop=True)
    
    X = df_clean.drop(columns=['stroke'])  # Features
    y = df_clean['stroke']  # Target variable
    
    # Perform stratified split
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    
    y_train = tf.one_hot(y_train.astype(np.int32), depth=1)#2)
    y_test = tf.one_hot(y_test.astype(np.int32), depth=1)#2)
    
    return x_train, y_train, x_test, y_test

def load_stroke_nprep():
    # see preprocessing in stroke_data_prepr.ipynb
    path = 'classes/Datasets/healthcare-dataset-stroke-data_new.csv'
    df = pd.read_csv(path)
    #df.drop(['id'], axis=1, inplace=True)

    cat_features = ['gender', 'hypertension', 'heart_disease', 'ever_married', 'work_type', 'Residence_type',
                    'smoking_status']
    cat_df = df[cat_features]
    cat_dummy = pd.get_dummies(cat_df)

    num_features = ['age', 'avg_glucose_level', 'bmi']
    num_df = df[num_features]
    imp_median = SimpleImputer(missing_values=np.nan, strategy='median')
    num_df = pd.DataFrame(imp_median.fit_transform(num_df), columns=num_df.columns)
    num_scaled = pd.DataFrame(MinMaxScaler().fit_transform(num_df), columns=num_df.columns)

    df_clean = pd.concat([num_scaled, cat_dummy, df['stroke']], axis=1).reset_index(drop=True)

    X = df_clean.drop(columns=['stroke'])  # Features
    y = df_clean['stroke']  # Target variable

    # Perform stratified split
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)


    y_train = tf.one_hot(y_train.astype(np.int32), depth=1)#2)
    y_test = tf.one_hot(y_test.astype(np.int32), depth=1)#2)

    return x_train, y_train, x_test, y_test

def load_medmnist():
    download=True
    data = 'pathmnist'
    info = INFO[data]
    size = 64# [28, 64, 128, 224]
    DataClass = getattr(medmnist, info['python_class'])

    # load the data
    train_dataset = DataClass(split='train', download=download, size=size, mmap_mode='r')
    test_dataset = DataClass(split='test', download=download, size=size, mmap_mode='r')

    # Initialize empty lists to store data and labels
    x_train, y_train = [], []
    x_test, y_test = [], []
    
    for data in itertools.zip_longest(train_dataset, test_dataset):
        train, test = data
        if train is not None:
            x_train.append(np.array(train[0]))
            y_train.append(train[1])
        if test is not None:
            x_test.append(np.array(test[0]))
            y_test.append(test[1])
    
    x_train, x_test = np.array(x_train)/ 255.0, np.array(x_test)/ 255.0
    y_train, y_test = np.squeeze(np.array(y_train)), np.squeeze(np.array(y_test))
    #x_train = x_train / 255.0
    #x_test = x_test / 255.0

    y_train = tf.one_hot(y_train.astype(np.int32), depth=len(info['label']))
    y_test = tf.one_hot(y_test.astype(np.int32), depth=len(info['label']))
    return x_train, y_train, x_test, y_test