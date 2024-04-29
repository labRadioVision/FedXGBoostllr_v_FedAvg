from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
import tensorflow.keras as tfk
import xgboost as xgb
import numpy as np
import json
from classes.Datasets.data_loader import load_mnist, load_stroke, load_stroke_nprep
from classes.Datasets.dataset_client import Dataset
from classes.Models.VanillaMLP import VanillaMLP
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
f = open('data/server/data_info.json',)
st = json.load(f)
loss = "binary_crossentropy"
metrics = ["accuracy"]
data_size = st['input_shape']
E_central = 20 # rounds centralized
E = 4 # local epochs
############################## Centralized performance,
# data are fused on the server, this is the classical distributed NN, privacy critical

model_centralized = VanillaMLP(data_size,1).return_model()  # create a new model
opt = tfk.optimizers.Adam(learning_rate=0.01, beta_1=0.5, beta_2=0.999)
model_centralized.compile(optimizer=opt, loss=loss, metrics=metrics)
x_train = np.array(x_train).astype('float32') # this is required for stroke no prep
model_centralized.fit(
    x_train, y_train, epochs=E_central, verbose=False
)  # train the model on the client data
x_valid = np.array(x_valid).astype('float32') # this is required for stroke no prep
model_centralized.evaluate(x_valid, y_valid)  # evaluate the global model

y_hat_c = model_centralized.predict(x_valid)
y_pred = y_hat_c >= 0.5 # binary estimator (CNN model has sigmoid output)

error_centr = accuracy(y_valid, y_pred)
cm = pd.DataFrame(confusion_matrix(y_valid, y_pred)).to_numpy()
TPR_centralized = cm[1,1]/(cm[1,0] + cm[1,1])
TNR_centralized = cm[0,0] / (cm[0,0] + cm[0,1])
print(f"Accuracy (Centralized), TPR, TNR: {100*error_centr :.5f} {100*TPR_centralized :.5f} {100*TNR_centralized :.5f}%")

#load saved model (just for test)
# xgb = joblib.load(checkpointpath1)
############################################

################################# INDIVIDUAL CLIENTS (NO FEDERATION)
# iid split (ext with sample/label/feature imbalance)
x_train_clients = []
y_train_clients = []
x_valid_clients = []
y_valid_clients = []
for k in range(num_clients):
    handle = Dataset(k)
    x_train_clients.append(handle.x_train_local)
    # y_train_mod = np.argmax(handle.y_train_local, axis=1) # convert from onehot
    y_train_clients.append(handle.y_train_local)
    x_valid_clients.append(handle.x_valid)
    # y_valid_mod = np.argmax(handle.y_valid, axis=1) # convert from onehot
    y_valid_clients.append(handle.y_valid)

datasets = tuple(zip(x_train_clients, y_train_clients))

######  Evaluate local nn models (no federation)
errors_clients = []
TPR_clients = []
TNR_clients = []
# model_clients = []
for c, (x_train, y_train) in enumerate(
        datasets
):  # extract the dataset for the current client
    model_client = VanillaMLP(data_size,1).return_model()  # create a new model
    opt = tfk.optimizers.Adam(learning_rate=0.01, beta_1=0.5, beta_2=0.999)
    model_client.compile(optimizer=opt, loss=loss, metrics=metrics)
    x_train = np.array(x_train).astype('float32') # this is required for stroke no prep
    model_client.fit(
        x_train, y_train, epochs=E_central, verbose=False
    )  # train the model on the client data
    # x_valid = np.array(x_valid).astype('float32') # this is required for stroke no prep
    model_client.evaluate(x_valid, y_valid)  # evaluate the global model
    y_hat_c = model_client.predict(x_valid)
    y_pred = y_hat_c >= 0.5 # binary estimator (CNN model has sigmoid output)
    error = accuracy(y_valid, y_pred)
    cm = pd.DataFrame(confusion_matrix(y_valid, y_pred)).to_numpy()
    TPR_isolated = cm[1,1]/(cm[1,0]+cm[1,1])
    TNR_isolated = cm[0,0] / (cm[0,0] + cm[0,1])
    print(f"Accuracy, TPR, TNR (Client {c}): {100*error :.5f} {100*TPR_isolated :.5f} {100*TNR_isolated :.5f}%")
    errors_clients.append(error)
    TPR_clients.append(TPR_isolated)
    TNR_clients.append(TNR_isolated)
    #model_clients.append(model_client)

############## FEDERATED AVERAGING VANILLA ###########################

R = 15  # global rounds
E = 10  # local epochs

models_clients = []  # list of models

model_global = VanillaMLP(data_size,1).return_model()  # create a new model
num_layers = len(model_global.get_weights())
opt = tfk.optimizers.Adam(learning_rate=0.01, beta_1=0.5, beta_2=0.999)
model_global.compile(optimizer=opt, loss=loss, metrics=metrics)
model_global.summary()
print(f"Round 0/{R}")  # init model
model_global.evaluate(x_valid, y_valid)

# FEDERATED LEARNING PROCESS (FEDAVG)
for r in range(R):  # for each round
    for c, (x_train_c, y_train_c) in enumerate(datasets):  # for each client
        print(f"Round {r + 1}/{R}, Client {c + 1}/{num_clients}")
        model_client = VanillaMLP(data_size,1).return_model()  # local model
        model_client.set_weights(model_global.get_weights())
        x_train_c = np.array(x_train_c).astype('float32') # required for stroke no prep
        opt = tfk.optimizers.Adam(learning_rate=0.01, beta_1=0.5, beta_2=0.999)
        model_client.compile(optimizer=opt, loss=loss, metrics=metrics)
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
    model_global.evaluate(x_valid, y_valid)  # evaluate the global model

################## Final testing on stroke data #####################
y_hat_g = model_global.predict(x_valid)
y_hat = y_hat_g >= 0.5 # binary estimator (CNN model has sigmoid output)

error_fed = accuracy(y_valid, y_hat)

# performance and confusion matrix
cm = confusion_matrix(y_valid, y_hat)
TPR_fed = cm[1,1] / (cm[1,0] + cm[1,1])
TNR_fed = cm[0,0] / (cm[0,0] + cm[0,1])

print(f"Accuracy (Centralized), TPR, TNR: {100*error_centr :.5f} {100*TPR_centralized :.5f} {100*TNR_centralized :.5f}%")
for c, error in enumerate(errors_clients):
    print(f"Accuracy, TPR, TNR: (Client {c}): {100*error :.5f} {100*TPR_clients[c] :.5f} {100*TNR_clients[c] :.5f}%")
print(f"Accuracy (Federated), TPR, TNR: {100*error_fed :.5f} {100*TPR_fed :.5f} {100*TNR_fed :.5f}%")

# saving results
checkpointpath = 'xgb_models/global_model_FedAvg_final.h5'
model_global.save(checkpointpath)
# joblib.dump(model_global, checkpointpath, compress=0)
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
    "results/fedavg_{}_alpha_{}_samples_{}_run_{}.mat".format(args.niid_type,args.alpha,args.samples,run), dict_1)
