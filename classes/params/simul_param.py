import tensorflow as tf

######  user selectable models, FL algorithms, architectures and optimizers  #########
TYPES_MODELS = {
    0: ['classes.models.CNN', 'CNN'],
    1: ['classes.models.VanillaMLP', 'VanillaMLP'],
    2: ['classes.models.Mobinet','Mobinet'],
    3: ['classes.models.VanillaLSTM', 'VanillaLSTM'],
    4: ['classes.models.1D_CNN', '1D_CNN']# Need something better
}  

ALG_MODULE = 'classes.algorithms'

TYPES_ARCHITECTURE = {
    0: ['PS_Synch'],            # PS-S/C-S
    1: ['PS_Asynch_PS_Synch'],  # PS-S/C-A
    2: ['PS_Asynch'],           # PS-A/C-A
    3: ['Consensus']
}

TYPES_OPTIMIZERS = {
    0: ['SGD'],
    1: ['ADAM']
}

architecture_id = 0                         
ARCHITECTURE = TYPES_ARCHITECTURE[architecture_id]                    # choose 0, 1,  2 or 3

model_id = 0
MODEL = TYPES_MODELS[model_id]                                 # choose 0, 9  | ex ['Classes.Models.CNN', 'CNN']
MODULE_MODEL_NAME = MODEL[0]                                                    # ex 'Classes.Models.CNN'
MODEL_NAME = MODEL[1]                                                           # ex 'CNN'

optimizer_id = 0
CHOSEN_OPTIMIZER = TYPES_OPTIMIZERS[optimizer_id]                  # choose 0, 1
if CHOSEN_OPTIMIZER[0] == 'SGD':
    OPTIMIZER = tf.keras.optimizers.SGD                   # Optimization algorithm
elif CHOSEN_OPTIMIZER[0] == 'ADAM': # Note: Might cause problems for some algorithms
    OPTIMIZER = tf.keras.optimizers.Adam
LR = 1e-2  # Learning rate

LOSS = tf.keras.losses.CategoricalCrossentropy()            # Loss to be minimized
METRICS = [tf.keras.metrics.CategoricalAccuracy(),
            tf.keras.metrics.Precision(),
            tf.keras.metrics.Recall(),
            tf.keras.metrics.AUC()]

