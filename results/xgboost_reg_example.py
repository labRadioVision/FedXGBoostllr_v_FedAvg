from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
import xgboost as xgb
import numpy as np
from Classes.Datasets.Stroke import Stroke_kaggle
from utils import get_trees_predictions_xgb, mse
from models import CNN

housing = fetch_california_housing()
x_train, x_valid, y_train, y_valid = train_test_split(
    housing.data, housing.target, test_size=0.33, random_state=11
)

training_set_per_device = 200  # NUMBER OF TRAINING SAMPLES PER DEVICE
start_index = 0
data_stroke = Stroke_kaggle(0, start_index, training_set_per_device)

# Set number of clients and number of trees per client
num_clients = 3  # K
trees_client = 20  # M
objective = "regression"

# Centralized performance
hyperparams = {
    "objective": "reg:squarederror",
    # Same number of trees as in the decentralized case
    "n_estimators": num_clients * trees_client,
    "max_depth": 6,
    "learning_rate": 0.1,
    "base_score": 0,
    "random_state": 34,
}

reg = xgb.XGBRegressor(**hyperparams)
reg.fit(x_train, y_train)
y_pred = reg.predict(x_valid)

error_centr = mse(y_valid, y_pred)
print(f"MSE (Centralized): {error_centr :.5f}")

# INDIVIDUAL CLIENTS
x_train_clients = np.array_split(x_train, num_clients)
y_train_clients = np.array_split(y_train, num_clients)

datasets = tuple(zip(x_train_clients, y_train_clients))

# Hyperparameters for each of the clients
hyperparams = {
    "objective": "reg:squarederror",
    "n_estimators": trees_client,
    "max_depth": 5,
    "learning_rate": 0.1,
    "base_score": 0,  # np.mean(y_train)
    "random_state": 34,
}

# Save the eneesmbles and evaluate them separately
XGB_models = []
errors_clients = []
for c, (x_train, y_train) in enumerate(
        datasets
):  # extract the dataset for the current client
    reg = xgb.XGBRegressor(**hyperparams)
    reg.fit(x_train, y_train)

    y_pred = reg.predict(x_valid)
    error = mse(y_valid, y_pred)
    print(f"MSE (Client {c}): {error :.5f}")
    errors_clients.append(error)

    XGB_models.append(reg)


# Create new data for 1D-CNN (XGB trees output-> 1D-CNN -> MSE)
# (X: output of the ensembles, Y: true label)

# Training
x_data_client_out = []
y_data_client_out = []

for c, (x_train, y_train) in enumerate(datasets):  # for each client
    print("Converting the data of client", c, 100 * "-")
    x_data_client_out.append(get_trees_predictions_xgb(x_train, objective, *XGB_models))
    y_data_client_out.append(y_train)

datasets_out = tuple(zip(x_data_client_out, y_data_client_out))

# Validation
xgb_valid_out = get_trees_predictions_xgb(x_valid, objective, *XGB_models)


# aggregator
R = 5  # global rounds
E = 2  # local epochs

# (num_clients, num_trees, num_channels)
params_cnn = (num_clients, trees_client, 16, objective)
models_clients = []  # list of models


model_global = CNN(*params_cnn)  # global model
num_layers = len(model_global.get_weights())

model_global.summary()
print(f"Round 0/{R}")  # init model
model_global.evaluate(xgb_valid_out, y_valid)

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
    for i in range(num_layers):  # aggregate the weights (fedavg, no memory)
        global_weights.append(
            np.sum([model.get_weights()[i] for model in models_clients], axis=0)
            / len(models_clients)
        )
    model_global.set_weights(global_weights)

    model_global.evaluate(xgb_valid_out, y_valid)  # evaluate the global model

# testing
y_hat_xgb = model_global.predict(xgb_valid_out)
error_fed = mse(y_valid, y_hat_xgb)

print(f"MSE (Centralized): {error_centr :.5f}")
for c, error in enumerate(errors_clients):
    print(f"MSE (Client {c}): {error :.5f}")
print(f"MSE (Federated): {error_fed :.5f}")