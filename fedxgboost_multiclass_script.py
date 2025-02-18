from sklearn import datasets
from sklearn.model_selection import train_test_split
import pickle
import argparse
import warnings


warnings.filterwarnings("ignore")
parser = argparse.ArgumentParser()
parser.add_argument('-reshape', default=False, help="reshape", type=bool)
parser.add_argument('-inputs', default="soft", help=" or binary ", type=str)
parser.add_argument('-num_clients', default=4, help="clients", type=int)

args = parser.parse_args()

num_classes = 4 # number of classes (>2)
n_features = 50 # number of features
n_redundant = 15 # redundant features
n_informative = n_features - n_redundant # informative features
test_size = 0.4 # fraction of data used for validation
training_samples = 1000

num_clients = args.num_clients  # number of FL clients
trees_client = 50  # number of xgboost trees per client
samples = round(training_samples/num_clients) # number of training examples per client
 
# load the dataset
with open(f"dataset/dataset_{num_classes}_redundant_{n_redundant}.pkl", 'rb') as f:
    x_train, x_valid,  y_train, y_valid = pickle.load(f)

import joblib
from utils import accuracy
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import xgboost as xgb
import numpy as np

# xgboost parameters (example)
hyperparams = {
    "objective": "multi:softmax",
    # Same number of trees as in the decentralized case
    "n_estimators": trees_client,
    "max_depth": 5,
    "learning_rate": 0.1,
    "base_score": 0.5,
    "random_state": 34,
}
if num_classes ==2:
    hyperparams["objective"] = "binary:logistic"
    
reg = xgb.XGBClassifier(**hyperparams)
reg.fit(x_train, y_train)
y_pred = reg.predict(x_valid)
accuracy_s = accuracy_score(y_valid, y_pred)

print(f"Accuracy: {accuracy_s:.2f}") 

cm = confusion_matrix(y_valid, y_pred)
# save and store the centralized model
checkpointpath1 = 'xgb_models/XGB_centralized_model.h5'
joblib.dump(reg, checkpointpath1, compress=0)


import os, json

print('Splitting IID')
local_size = samples  # uniform split, and a assign 'samples' samples
# split the training dataset and create folders in data/client_#i/train
for i in range(num_clients):
    dir = f'data/client_{i}/train/' # create a folder with the local data for the client
    os.makedirs(dir, exist_ok=True)
    start_index = i * local_size
    end_index = (i + 1) * local_size
    x_part = x_train[start_index:end_index]
    y_part = y_train[start_index:end_index]
        
    print('Client {} | Samples {}'.format(i, len(y_part)))
    np.save(dir + f'x_train.npy', x_part) # creating directories for train and validation
    np.save(dir + f'y_train.npy', y_part)
print(f'Saved train data')

# split the validation dataset and create folders in data/client_#i/valid
local_size = len(x_valid) // num_clients # validation data uniformly distributed across clients (other options are also possible) 
# local_size = len(x_valid)
for i in range(num_clients):
    dir = f'data/client_{i}/valid/' # create a folder with the local data for the client
    os.makedirs(dir, exist_ok=True)
    start_index = i * local_size
    end_index = (i + 1) * local_size
    #x_part = x_valid[start_index:end_index] # uniform split of the validation set
    #y_part = y_valid[start_index:end_index]
    x_part = x_valid # all the clients have the same validation set (for fair comparison)
    y_part = y_valid
        
    print('Client {} | Samples {}'.format(i, len(y_part)))
    np.save(dir + f'x_valid.npy', x_part) # saving
    np.save(dir + f'y_valid.npy', y_part)
print(f'Saved validation data')


x_train_clients = []
y_train_clients = []
x_valid_clients = []
y_valid_clients = []

# create train and valid datasets for all clients
for k in range(num_clients):
    x_train_clients.append(np.load(f'data/client_{k}/train/x_train.npy', allow_pickle=True))
    y_train_clients.append(np.load(f'data/client_{k}/train/y_train.npy'))
    x_valid_clients.append(np.load(f'data/client_{k}/valid/x_valid.npy', allow_pickle=True))
    y_valid_clients.append(np.load(f'data/client_{k}/valid/y_valid.npy'))

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
if num_classes ==2:
    hyperparams["objective"] = "binary:logistic"
    
errors_clients = []

for c, (x_train, y_train) in enumerate(
        datasets
):  # extract the dataset for the current client
    reg = xgb.XGBClassifier(**hyperparams) # train the classifier
    reg.fit(x_train, y_train)
    # save model
    checkpointpath = 'xgb_models/XGB_client_model_{}.h5'.format(c)
    joblib.dump(reg, checkpointpath, compress=0)
    
    # full performance tests (accuracy and confusion matrix)
    y_pred = reg.predict(x_valid)
    error = accuracy_score(y_valid, y_pred)
    cm = confusion_matrix(y_valid, y_pred)
    print(f"xgboost classifier local model accuracy, (Client {c}): {100*error :.5f}%")
    errors_clients.append(error)

    ################################# INDIVIDUAL CLIENTS XGBOOST REGRESSION MODEL TRAINING)
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
if num_classes ==2:
    hyperparams["objective"] = "binary:logistic"
    
for c, (x_train, y_train) in enumerate(
        datasets
):  # extract the dataset for the current client
    reg = OneVsRestClassifier(xgb.XGBClassifier(**hyperparams)) # regression model training
    reg.fit(x_train, y_train)
    # save model
    # note: there are two ways to save a one vs rest classifier model, either using joblib or pickle, json is not supported
    checkpointpath = 'xgb_models/XGB_client_model_fed_{}.h5'.format(c)
    joblib.dump(reg, checkpointpath, compress=0)
  
# test of the one vs all local model (compared with the previous case)
    y_pred = reg.predict(x_valid)
    error = accuracy_score(y_valid, y_pred)
    cm = confusion_matrix(y_valid, y_pred)
    print(f"One vs all local model accuracy, (Client {c}): {100*error :.5f}%")


from utils import get_trees_predictions_xgb

reshape_enabled = args.reshape
#inputs_obj = "binary" 

if num_classes == 2:
    reshape_enabled = False # always disabled for 2 classes (binary problem)

inputs_obj = args.inputs
# other options: 
# objective = "soft" # applies a tanh activation to the xgboost tree soft outputs  
# objective = "binary" # outputs of xgboost trees are binarized, 

# load all xgboost models and prepare the data
XGB_models = []
for c in range(num_clients):
    if num_classes == 2:
        checkpointpath = 'xgb_models/XGB_client_model_{}.h5'.format(c)
    else:
        checkpointpath1 = 'xgb_models/XGB_client_model_fed_{}.h5'.format(c)
    xgb = joblib.load(checkpointpath1)
    if num_classes == 2:
        XGB_models.append(xgb)
    else:
        classifiers = xgb.estimators_
        for q in range(num_classes):
            XGB_models.append(classifiers [q])

x_xgb_trees_out = []
y_xgb_trees_out = []
for c, (x_train, y_train) in enumerate(datasets):  # for each client
    print("Converting the data of client", c, 100 * "-")
    # XGB trees outputs (for all XGBoost trees!) corresponding to training data of client c
    # NOTE: numclasses is needed to clip the inputs
    x_xgb_trees_out.append(get_trees_predictions_xgb(x_train, inputs_obj, *XGB_models, numclasses=num_classes, reshape_enabled=reshape_enabled)) 
    if num_classes == 2:
        categorical_labels = y_train
    else:
        categorical_labels = np.squeeze(np.eye(num_classes)[y_train.reshape(-1)])
    y_xgb_trees_out.append(categorical_labels) # true labels now categorical

datasets_out = tuple(zip(x_xgb_trees_out, y_xgb_trees_out)) # dataset_out is the new federated dataset input to 1D-CNN (XGB trees output-> 1D-CNN -> accuracy)

# Validation data

xgb_valid_out = get_trees_predictions_xgb(x_valid, inputs_obj, *XGB_models, numclasses=num_classes, reshape_enabled=reshape_enabled) # XGB trees outputs corresponding to validation data: to simplify the reasoning, we apply same validation set for all (other options are also feasible)

from models import CNN_mc # check the model in models.py

filters = 32 # convolutional filters (32 ok, >32 too large, depends on tree structures) TO BE OPTIMIZED
if reshape_enabled:
    filter_size = trees_client
else:
    if num_classes == 2:
        filter_size = trees_client # CNN filter size equal to the number of trees per client 
    else:
        filter_size = trees_client * num_classes # CNN filter size equal to the number of trees per client * number of classes (if > 2)

params_cnn = (num_clients, filter_size, trees_client, filters, num_classes)
models_clients = []  # list of models

model_global = CNN_mc(*params_cnn)  # global model
num_layers = len(model_global.get_weights())

model_global.summary()

R = 25  # global FL rounds
E = 10  # local epochs

print(f"Round 0/{R}")  # init model

# sets labels to categorical
if num_classes == 2:
    categorical_y_valid = y_valid
else:
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

    import scipy.io as sio

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
checkpointpath = 'xgb_models/XGB_federated_model_regression_multiclass.keras'
model_global.save(checkpointpath)
dict_1 = {
    "Accuracy_centralized": accuracy_s,
    "Accuracy_clients": errors_clients,
    "Accuracy_federated": accuracy_fed
}
sio.savemat(
    "results/fedXGboost_{}_features_{}_redundant_{}_classes_{}_clients_{}_trees_client_{}_train_samples_{}_reshape_{}_objective{}.mat".format('iid',n_features,n_redundant,num_classes, num_clients, trees_client, samples, reshape_enabled, inputs_obj), dict_1)
