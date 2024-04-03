############ NUMBER OF CLIENTS AND SELECTED CLIENTS (BY THE PS) PER ROUND
NUM_CLIENTS = 6 # for now controlled in launcher
# Clients at each Federated_round
CLIENTS_SELECTED = 3
assert CLIENTS_SELECTED <= NUM_CLIENTS

######### configuration parameters
ALL_DATA = False  # use all training data vs manually select samples per device (SEE PERCENTAGE IN CLASS DATASET)
FEDERATION = True  # enable federation
SHED_TYPE = 'Robin' # 'Robin' - round robin, 'Rand' - random selection
####################################

BATCH_SIZE = 32                                                 # B
NUM_BATCHES = 10
NUM_ROUNDS = 50  # 600 (FL rounds)
NUM_EPOCHS = 4   # 2 MNIST  (Number of local rounds)

PATIENCE = NUM_ROUNDS                                           # Early stopping in Federated Rounds
MAX_MODEL_SIZE = 250 * 10**6
#########################THIRD

# Consensus parameters
EPSILON = 1    # 0.3 MNIST                                                  # Consensus step size
Q = 0.99         # 0.99 MNIST                                                # Hyperparameter in MEWMA for gradients
Beta = 1# 10**(-3) #200                                         # Mixing weights for the gradients
Gossip = 0                                                      # 1 if Gossip is enabled
# COORD_SLEEP_TIME = 497*1/16                                            # [s]
CLIENT_WAIT_TIME_CONSENSUS = 6

# PS parameters
EPSILON_GLOBAL = 1                                        # Hyperparameter in MEWMA for global_model
SERVER_SLEEP_TIME = 2  # 2 MNIST                                           # In case of Asynchronous Server [s]

# Learner parameter
CLIENT_WAIT_TIME = 4


# Algorithm parameters 
ALPHA_ADP = 3 # Fed Adp parameter
MU = 0.1# Fed Prox parameter
ALPHA_DYN = 0.01# Fed Dyn parameter