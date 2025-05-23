# importing necessary libraries
from sklearn import datasets
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import xgboost as xgb
import tensorflow as tf
from tensorflow.keras import backend as K
import numpy as np
from classes.Datasets.dataset_client import Dataset
from classes.Datasets.data_loader import load_random
from classes.params import simul_param, fl_param
from utils import get_trees_predictions_xgb, accuracy, load_unsurance
from models import CNN_mc
from sklearn.metrics import accuracy_score
import pandas as pd
import scipy.io as sio
import joblib
import argparse
import warnings

def round_to_range(y_pred, min_value, max_value):
  rounded_vector = np.rint(y_pred)  # Round to nearest integer
  rounded_vector = np.clip(rounded_vector, min_value, max_value)  # Clip values to the range
  return rounded_vector

# dividing X, y into train and test data
# x_train, x_valid,  y_train, y_valid = train_test_split(X, y, test_size=0.5, random_state=42)
x_train, y_train, x_valid, y_valid = load_random()
num_classes = 4
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
    errors_clients.append(error)
    #TPR_clients.append(TPR_isolated)
    #TNR_clients.append(TNR_isolated)

################################# INDIVIDUAL CLIENTS TRAINING FOR FEDERATION only / REGRESSION)
from sklearn.multiclass import OneVsRestClassifier

# Hyperparameters for each of the clients
hyperparams = {
    # "objective": "reg:squarederror",
    "objective": "binary:logistic",
    "n_estimators": trees_client,
    "max_depth": 5,
    "learning_rate": 0.1,
    "base_score": 0.5,  # np.mean(y_train)
    "random_state": 34,
}

for c, (x_train, y_train) in enumerate(
        datasets
):  # extract the dataset for the current client
    reg = OneVsRestClassifier(xgb.XGBClassifier(**hyperparams)) # regression model training
    reg.fit(x_train, y_train)
    # save model
    checkpointpath = 'xgb_models/XGB_client_model_fed_{}.h5'.format(c)
    joblib.dump(reg, checkpointpath, compress=0)
  
# test of the one vs all local model (compared with the previous case)
    y_pred = reg.predict(x_valid)
    error = accuracy_score(y_valid, y_pred)
    cm = confusion_matrix(y_valid, y_pred)
    print(f"One vs all local model accuracy, (Client {c}): {100*error :.5f}%")

############## FEDERATED XGBOOST ###########################
# Create FIRST a new data for 1D-CNN (XGB trees output-> 1D-CNN -> accuracy)
# (X: output of the ensembles, Y: true label)

# all clients xgboost model must be shared before starting FL process, can be loaded from a shared folder
# load all xgboost models and prepare the data
XGB_models = []
for c in range(num_clients):
    checkpointpath1 = 'xgb_models/XGB_client_model_fed_{}.h5'.format(c)
    xgb = joblib.load(checkpointpath1)
    classifiers = xgb.estimators_
    for q in range(num_classes):
        XGB_models.append(classifiers [q])

# prepare the new dataset for training
objective = "binary" 
objective = "soft" # applies a tanh activation to the xgboost tree outputs 
# works also with "binary" where outputs of xgboost trees are binarized, binary option seems working slightly better but unclear
reshape_enabled = True # disable or enable the reshaping of xgboost outputs

if reshape_enabled:
    filter_size = trees_client
else:
    filter_size = trees_client * num_classes # CNN filter size equal to the number of trees per client * number of classes (if > 2)

filters = 32 # convolutional filters (16, 32 ok, >32 too large, depends on tree structures) TO BE OPTIMIZED

x_xgb_trees_out = []
y_xgb_trees_out = []
for c, (x_train, y_train) in enumerate(datasets):  # for each client
    print("Converting the data of client", c, 100 * "-")
    # XGB trees outputs (for all XGBoost trees!) corresponding to training data of client c
    # NOTE: numclasses is needed to clip the inputs
    x_xgb_trees_out.append(get_trees_predictions_xgb(x_train, objective, *XGB_models, numclasses=num_classes, reshape_enabled=reshape_enabled)) 
    categorical_labels = np.squeeze(np.eye(num_classes)[y_train.reshape(-1)])
    y_xgb_trees_out.append(categorical_labels) # true labels now categorical

datasets_out = tuple(zip(x_xgb_trees_out, y_xgb_trees_out)) # dataset_out is the new federated dataset input to 1D-CNN (XGB trees output-> 1D-CNN -> accuracy)

# Validation data

xgb_valid_out = get_trees_predictions_xgb(x_valid, objective, *XGB_models, numclasses=num_classes, reshape_enabled=reshape_enabled) # XGB trees outputs corresponding to validation data: to simplify the reasoning, we apply same validation set for all (other options are also feasible)


params_cnn = (num_clients, filter_size, trees_client, filters, num_classes)
models_clients = []  # list of models

model_global = CNN_mc(*params_cnn)  # global model
num_layers = len(model_global.get_weights())

model_global.summary()


############# FEDXGBOOST aggregator
R = 25  # global FL rounds
E = 10  # local epochs

print(f"Round 0/{R}")  # init model

# sets labels to categorical
categorical_y_valid = np.squeeze(np.eye(num_classes)[y_valid.reshape(-1)])

for r in range(R):  # for each round
    
    # update phase for each client
    for c, (x_train_c, y_train_c) in enumerate(datasets_out):  
        print(f"Round {r + 1}/{R}, Client {c + 1}/{num_clients}")
        model_client = CNN_mc(*params_cnn)  # create a new model
        # set global weights (no memory of prev local weights)
        model_client.set_weights(model_global.get_weights())  
        # update phase
        model_client.fit(
            x_train_c, y_train_c, epochs=E, verbose=False
        )  # train the model on the client data
        models_clients.append(model_client)  # save the model
    
    # aggregation phase
    global_weights = []
    for i in range(num_layers):  # aggregate the weights, no memory of prev global weights
        global_weights.append(
            np.sum([model.get_weights()[i] for model in models_clients], axis=0)
            / len(models_clients)
        )
    model_global.set_weights(global_weights)

    model_global.evaluate(xgb_valid_out, categorical_y_valid)  # evaluate the global model

################# Final testing on data #####################
y_hat_xgb = model_global.predict(xgb_valid_out)
y_hat_xgb_cont = np.argmax(y_hat_xgb, axis=1)
accuracy_fed = accuracy_score(y_valid, y_hat_xgb_cont)
cm = confusion_matrix(y_valid, y_hat_xgb_cont)

# performance and confusion matrix

print(f"Accuracy (Centralized): {accuracy_s :.2f}")
for c, error in enumerate(errors_clients):
    print(f"Accuracy (Client {c}): {error :.2f}")
print(f"Accuracy (Federated): {accuracy_fed :.2f}")

# saving results
checkpointpath = 'xgb_models/XGB_federated_model_regression_multiclass.h5'
model_global.save(checkpointpath)

# saving results
checkpointpath = 'xgb_models/XGB_federated_model_regression_multiclass.h5'
model_global.save(checkpointpath)
# joblib.dump(model_global, checkpointpath, compress=0)
dict_1 = {"Accuracy_centralized": accuracy_s,
          "Accuracy_clients": errors_clients,
          "Accuracy_federation": accuracy_fed,
          }
sio.savemat(
    "results/fedXGboost_regression_multiclass.mat", dict_1)
