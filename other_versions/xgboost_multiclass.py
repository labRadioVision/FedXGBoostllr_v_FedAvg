# importing necessary libraries
from sklearn import datasets
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import xgboost as xgb
import tensorflow as tf
import numpy as np
from classes.Datasets.dataset_client import Dataset
from classes.Datasets.data_loader import load_iris, load_random
from classes.params import simul_param, fl_param
from utils import get_trees_predictions_xgb, accuracy, load_unsurance
from models import CNN
from sklearn.metrics import accuracy_score
import pandas as pd
import scipy.io as sio
import joblib
import argparse
import warnings
# loading the iris dataset
#iris = datasets.load_iris()

# X -> features, y -> label
#X = iris.data
#y = iris.target

# dividing X, y into train and test data
# x_train, x_valid,  y_train, y_valid = train_test_split(X, y, test_size=0.5, random_state=42)
x_train, y_train, x_valid, y_valid = load_random()
num_clients = fl_param.NUM_CLIENTS  # K
trees_client = 10 # M
# objective = "binary"
############################## Centralized performance,
# data are fused on the server, this is the classical distributed xboost, privacy critical
hyperparams = {
    "objective": "multi:softmax",
    # Same number of trees as in the decentralized case
    "n_estimators": num_clients * trees_client,
    "max_depth": 5,
    "learning_rate": 0.1,
    "base_score": 0.5,
    "random_state": 34,
}

reg = xgb.XGBClassifier(**hyperparams)
reg.fit(x_train, y_train)
y_pred = reg.predict(x_valid)
accuracy_s = accuracy_score(y_valid, y_pred)

print(f"Accuracy: {accuracy_s:.2f}") 

cm = pd.DataFrame(confusion_matrix(y_valid, y_pred)).to_numpy()

# save and store the centralized model
checkpointpath1 = 'xgb_models/XGB_centralized_model.h5'
joblib.dump(reg, checkpointpath1, compress=0)
# training a DescisionTreeClassifier

################################# INDIVIDUAL CLIENTS (NO FEDERATION)
# iid split (ext with sample/label/feature imbalance)
from classes.Datasets.dataset_client import Dataset
from classes.Datasets.data_partitioner import split_iid_sim
import os, json
# or run python -m classes.Datasets.data_generator.py to get a data distribution
samples = 100
niid_type = 'iid'
alpha = 1

print('Splitting IID')

# split the training dataset and create folders in data/client_#i/train
split_iid_sim(x_train, y_train, samples, num_clients, type='train')

# split the validation dataset and create folders in data/client_#i/valid
split_iid_sim(x_valid, y_valid, samples, num_clients, type='valid')

## optional save data info to json for PS only
# n_classes = np.unique(y_valid, axis=0).shape[0] if np.unique(y_valid, axis=0).shape[0]>2 else 1
# data_info = {
#        'input_shape': x_train.shape[1:],
#        'num_classes': n_classes, #np.unique(y_valid, axis=0).shape[0],
#        'data': data,
#        'niid_type': niid_type,
#        'alpha': alpha
#    }
# optional save data/server/
# dir = "data/server/"
# os.makedirs(dir, exist_ok=True)
# with open(dir + "data_info.json", "w") as outfile:
#    json.dump(data_info, outfile)

x_train_clients = []
y_train_clients = []
x_valid_clients = []
y_valid_clients = []

# create train and valid datasets for all clients
for k in range(num_clients):
    handle = Dataset(k) # get an handle to training dataset of client k
    x_train_clients.append(handle.x_train_local)
    y_train_clients.append(handle.y_train_local)
    x_valid_clients.append(handle.x_valid)
    y_valid_clients.append(handle.y_valid)

datasets = tuple(zip(x_train_clients, y_train_clients))

# Hyperparameters for each of the clients
hyperparams = {
    "objective": "multi:softmax",
    "n_estimators": trees_client,
    "max_depth": 5,
    "learning_rate": 0.1,
    "base_score": 0.5,  # np.mean(y_train)
    "random_state": 34,
}

errors_clients = []
TPR_clients = []
TNR_clients = []
for c, (x_train, y_train) in enumerate(
        datasets
):  # extract the dataset for the current client
    reg = xgb.XGBClassifier(**hyperparams)
    reg.fit(x_train, y_train)
    # save model
    checkpointpath = 'xgb_models/XGB_client_model_{}.h5'.format(c)
    joblib.dump(reg, checkpointpath, compress=0)
    
    # full performance tests (accuracy and confusion matrix)
    y_pred = reg.predict(x_valid)
    error = accuracy_score(y_valid, y_pred)
    cm = confusion_matrix(y_valid, y_pred)
    print(f"Accuracy, (Client {c}): {100*error :.5f}%")

    #TPR_isolated = cm[1,1] / (cm[1,0] + cm[1,1])
    #TNR_isolated = cm[0,0] / (cm[0,0] + cm[0,1])
    #print(f"Accuracy, TPR, TNR (Client {c}): {100*error :.5f} {100*TPR_isolated :.5f} {100*TNR_isolated :.5f}%")
    #errors_clients.append(error)
    #TPR_clients.append(TPR_isolated)
    #TNR_clients.append(TNR_isolated)

