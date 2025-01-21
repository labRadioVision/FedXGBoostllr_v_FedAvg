# importing necessary libraries
from sklearn import datasets
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import xgboost as xgb
import tensorflow as tf
import numpy as np
from classes.Datasets.dataset_client import Dataset
from classes.Datasets.data_loader import load_random_regression
from classes.params import simul_param, fl_param
from utils import get_trees_predictions_xgb, accuracy, load_unsurance
from models import CNN
from sklearn.metrics import mean_squared_error, r2_score
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
x_train, y_train, x_valid, y_valid = load_random_regression()
num_clients = fl_param.NUM_CLIENTS  # K
trees_client = 10 # M
# objective = "binary"
############################## Centralized performance,
# data are fused on the server, this is the classical distributed xboost, privacy critical
hyperparams = {
    "objective": "reg:squarederror",
    # Same number of trees as in the decentralized case
    "n_estimators": num_clients * trees_client,
    "max_depth": 5,
    "learning_rate": 0.1,
    "base_score": 0.5,
    "random_state": 34,
}

reg = xgb.XGBRegressor(**hyperparams)
reg.fit(x_train, y_train)
y_pred = reg.predict(x_valid)
accuracy_s = mean_squared_error(y_valid, y_pred)

print(f"Accuracy: {accuracy_s:.2f}") 

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
    "objective": "reg:squarederror",
    "n_estimators": trees_client,
    "max_depth": 5,
    "learning_rate": 0.1,
    "base_score": 0.5,  # np.mean(y_train)
    "random_state": 34,
}

errors_clients = []
for c, (x_train, y_train) in enumerate(
        datasets
):  # extract the dataset for the current client
    reg = xgb.XGBRegressor(**hyperparams)
    reg.fit(x_train, y_train)
    # save model
    checkpointpath = 'xgb_models/XGB_client_model_{}.h5'.format(c)
    joblib.dump(reg, checkpointpath, compress=0)
    
    # full performance tests (accuracy and confusion matrix)
    y_pred = reg.predict(x_valid)
    error = mean_squared_error(y_valid, y_pred)
    print(f"Accuracy, (Client {c}): {100*error :.5f}%")

    #TPR_isolated = cm[1,1] / (cm[1,0] + cm[1,1])
    #TNR_isolated = cm[0,0] / (cm[0,0] + cm[0,1])
    #print(f"Accuracy, TPR, TNR (Client {c}): {100*error :.5f} {100*TPR_isolated :.5f} {100*TNR_isolated :.5f}%")
    #errors_clients.append(error)
    #TPR_clients.append(TPR_isolated)
    #TNR_clients.append(TNR_isolated)

############## FEDERATED XGBOOST TO BE COMPLETED###########################
# Create FIRST a new data for 1D-CNN (XGB trees output-> 1D-CNN -> accuracy)
# (X: output of the ensembles, Y: true label)

# all clients xgboost model must be shared before starting FL process, can be loaded from a shared folder
# load all models (or use XGB_models)
XGB_models = []
for c in range(num_clients):
    checkpointpath1 = 'xgb_models/XGB_client_model_{}.h5'.format(c)
    xgb = joblib.load(checkpointpath1)
    XGB_models.append(xgb)

# Training
objective = "regression"
x_data_client_out = []
y_data_client_out = []
# xgb_valid_out = []
for c, (x_train, y_train) in enumerate(datasets):  # for each client
    print("Converting the data of client", c, 100 * "-")
    x_data_client_out.append(get_trees_predictions_xgb(x_train, objective, *XGB_models))
    y_data_client_out.append(y_train)

datasets_out = tuple(zip(x_data_client_out, y_data_client_out))

# Validation
xgb_valid_out = get_trees_predictions_xgb(x_valid, objective, *XGB_models)



############# FEDXGBOOST aggregator
R = 15  # global rounds
E = 10  # local epochs
filters = 16 # convolutional filters (16 32 ok, not too large)
filter_size = trees_client # CNN filter size must be equal to the number of trees per client

params_cnn = (num_clients, filter_size, filters, objective)
models_clients = []  # list of models

model_global = CNN(*params_cnn)  # global model
num_layers = len(model_global.get_weights())

model_global.summary()
print(f"Round 0/{R}")  # init model
model_global.evaluate(xgb_valid_out, y_valid)

# FEDERATED LEARNING PROCESS (FEDAVG)
for r in range(R):  # for each round
    for c, (x_train_c, y_train_c) in enumerate(datasets_out):  # for each client
        print(f"Round {r + 1}/{R}, Client {c + 1}/{num_clients}")
        model_client = CNN(*params_cnn)  # create a new model
        model_client.set_weights(model_global.get_weights())

        model_client.fit(
            x_train_c, y_train_c, epochs=E, verbose=False
        )  # train the model on the client data
        models_clients.append(model_client)  # save the model

    global_weights = []
    for i in range(num_layers):  # aggregate the weights
        global_weights.append(
            np.sum([model.get_weights()[i] for model in models_clients], axis=0)
            / len(models_clients)
        )
    model_global.set_weights(global_weights)

    model_global.evaluate(xgb_valid_out, y_valid)  # evaluate the global model