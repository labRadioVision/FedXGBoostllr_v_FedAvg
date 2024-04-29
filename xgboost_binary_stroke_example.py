from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
import xgboost as xgb
import tensorflow as tf
import numpy as np
from classes.Datasets.dataset_client import Dataset
from classes.Datasets.data_loader import load_mnist, load_stroke, load_stroke_nprep
from classes.params import simul_param, fl_param
from utils import get_trees_predictions_xgb, accuracy, load_unsurance
from models import CNN
from sklearn.metrics import confusion_matrix, accuracy_score
import pandas as pd
import scipy.io as sio
import joblib
import argparse
import warnings

warnings.filterwarnings("ignore")
parser = argparse.ArgumentParser()
parser.add_argument("-run", default=0, help="run number", type=int)
parser.add_argument('-samples', default=70, help=" max samples per device", type=int)
parser.add_argument("-niid_type", choices=['iid', 'label', 'sample', 'feature'], default='iid', help="Heterogeneity type", type=str)
parser.add_argument('-alpha', default=1, help=" alpha for non-iid (sigma for noise)", type=float) # small alpha for non-IID
args = parser.parse_args()

# choose the dataset
# data = "kaggle_stroke" # stroke data with SMOTE
data = "kaggle_stroke_nprep" # stroke data without SMOTE
run = args.run

# load the data (only for centralized perf)
if data == "kaggle_stroke":
    x_train, y_train, x_valid, y_valid = load_stroke()
elif data == "kaggle_stroke_nprep":
    x_train, y_train, x_valid, y_valid = load_stroke_nprep()
# Set number of clients and number of xgboost trees per client
num_clients = fl_param.NUM_CLIENTS  # K
trees_client = 15  # M
objective = "binary"

############################## Centralized performance,
# data are fused on the server, this is the classical distributed xboost, privacy critical
hyperparams = {
    "objective": "binary:logistic",
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

error_centr = accuracy(y_valid, y_pred)
cm = pd.DataFrame(confusion_matrix(y_valid, y_pred)).to_numpy()
TPR_centralized = cm[1,1]/(cm[1,0] + cm[1,1])
TNR_centralized = cm[0,0] / (cm[0,0] + cm[0,1])
print(f"Accuracy (Centralized), TPR, TNR: {100*error_centr :.5f} {100*TPR_centralized :.5f} {100*TNR_centralized :.5f}%")
# save and store the centralized model
checkpointpath1 = 'xgb_models/XGB_centralized_model.h5'
joblib.dump(reg, checkpointpath1, compress=0)

# load saved model (just for test)
# xgb = joblib.load(checkpointpath1)
############################################

################################# INDIVIDUAL CLIENTS (NO FEDERATION)
# iid split (ext with sample/label/feature imbalance)
num_sample = 200 # not used
x_train_clients = []
y_train_clients = []
x_valid_clients = []
y_valid_clients = []
for k in range(num_clients):
    handle = Dataset(k)
    x_train_clients.append(handle.x_train_local)
    y_train_clients.append(handle.y_train_local)
    x_valid_clients.append(handle.x_valid)
    y_valid_clients.append(handle.y_valid)

datasets = tuple(zip(x_train_clients, y_train_clients))

# Hyperparameters for each of the clients
hyperparams = {
    "objective": "binary:logistic",
    "n_estimators": trees_client,
    "max_depth": 5,
    "learning_rate": 0.1,
    "base_score": 0.5,  # np.mean(y_train)
    "random_state": 34,
}

# Save the ensembles and evaluate them separately (no federation)
# XGB_models = []
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

    error = accuracy(y_valid, y_pred)
    cm = pd.DataFrame(confusion_matrix(y_valid, y_pred)).to_numpy()
    TPR_isolated = cm[1,1] / (cm[1,0] + cm[1,1])
    TNR_isolated = cm[0,0] / (cm[0,0] + cm[0,1])
    print(f"Accuracy, TPR, TNR (Client {c}): {100*error :.5f} {100*TPR_isolated :.5f} {100*TNR_isolated :.5f}%")
    errors_clients.append(error)
    TPR_clients.append(TPR_isolated)
    TNR_clients.append(TNR_isolated)
    # XGB_models.append(reg)

############## FEDERATED XGBOOST ###########################
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

################## Final testing on stroke data #####################
y_hat_xgbb = model_global.predict(xgb_valid_out)
y_hat_xgb = y_hat_xgbb >= 0.5 # binary estimator (CNN model has sigmoid output)

error_fed = accuracy(y_valid, y_hat_xgb)

# performance and confusion matrix
cm = pd.DataFrame(confusion_matrix(y_valid, y_hat_xgb)).to_numpy()
TPR_fed = cm[1,1] / (cm[1,0] + cm[1,1])
TNR_fed = cm[0,0] / (cm[0,0] + cm[0,1])


print(f"Accuracy (Centralized), TPR, TNR: {100*error_centr :.5f} {100*TPR_centralized :.5f} {100*TNR_centralized :.5f}%")
for c, error in enumerate(errors_clients):
    print(f"Accuracy, TPR, TNR: (Client {c}): {100*error :.5f} {100*TPR_clients[c] :.5f} {100*TNR_clients[c] :.5f}%")
print(f"Accuracy (Federated), TPR, TNR: {100*error_fed :.5f} {100*TPR_fed :.5f} {100*TNR_fed :.5f}%")

# saving results
checkpointpath = 'xgb_models/global_model_FedAvg_final.h5'
joblib.dump(model_global, checkpointpath, compress=0)
dict_1 = {"Accuracy_centralized": error_centr,
          "TPR_centralized":  TPR_centralized,
          "TNR_centralized":  TNR_centralized,
          "Accuracy_clients": errors_clients,
          "TPR_clients": TPR_clients,
          "TNR_clients": TNR_clients,
          "Accuracy_federation": error_fed,
          "TPR_federation": TPR_fed,
          "TNR_federation": TNR_fed,

          }
sio.savemat(
    "results/fedXGboost_{}_alpha_{}_samples_{}_run_{}.mat".format(args.niid_type,args.alpha,args.samples,run), dict_1)
