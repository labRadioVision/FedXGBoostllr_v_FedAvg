# Define 1D-CNN
import tensorflow.keras as tfk


def CNN(num_clients, trees_client, n_channels, objective, n_classes=None):
    # Define 1D-CNN
    model = tfk.models.Sequential()
    model.add(
        tfk.layers.Conv1D(
            n_channels,
            kernel_size=trees_client,
            strides=trees_client,
            activation="relu",
            input_shape=(num_clients * trees_client, 1),
        )
    )

    model.add(tfk.layers.Flatten())
    model.add(tfk.layers.Dense(n_channels * num_clients, activation="relu"))

    # Output layer
    if objective == "binary":
        model.add(tfk.layers.Dense(1, activation="sigmoid"))
        loss = "binary_crossentropy"
        metrics = ["accuracy"]
    elif objective == "regression":
        model.add(tfk.layers.Dense(1, activation="linear"))
        loss = "mse"
        metrics = [None]
    elif objective == "multiclass":
        model.add(tfk.layers.Dense(n_classes, activation="softmax"))
        loss = "categorical_crossentropy"
        metrics = ["accuracy"]
    # Compile the model

    opt = tfk.optimizers.Adam(learning_rate=0.01, beta_1=0.5, beta_2=0.999)
    model.compile(optimizer=opt, loss=loss, metrics=metrics)
    return model


def SimpleNN(num_clients, trees_client, objective, lbd=0, n_classes=None):
    # Define 1D-CNN
    model = tfk.models.Sequential()
    model.add(tfk.layers.Input(shape=(num_clients * trees_client,)))

    # Output layer
    if objective == "binary":
        model.add(
            tfk.layers.Dense(
                1, activation="sigmoid", kernel_regularizer=tfk.regularizers.l1(lbd)
            ),
        )  # Lasso
        loss = "binary_crossentropy"
        metrics = ["accuracy"]
    elif objective == "regression":
        model.add(tfk.layers.Dense(1, activation="linear"))
        loss = "mse"
        metrics = [None]
    elif objective == "multiclass":
        model.add(tfk.layers.Dense(n_classes, activation="softmax"))
        loss = "categorical_crossentropy"
        metrics = ["accuracy"]

    # Compile the model
    opt = tfk.optimizers.Adam(learning_rate=0.01, beta_1=0.5, beta_2=0.999)
    model.compile(optimizer=opt, loss=loss, metrics=metrics)
    return model
