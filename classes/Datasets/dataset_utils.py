import numpy as np
from datetime import timedelta, datetime
from scipy.stats import wasserstein_distance

    
def kl_divergence(y_train_loc, y_train_glob):
    p = np.sum(y_train_loc, axis=0)/len(y_train_loc)
    q = np.sum(y_train_glob, axis=0)/len(y_train_glob)
    
    kl_divergence = np.sum(np.where(p != 0, p * np.log(p / q), 0))  # entropy(p, q)
    return kl_divergence

def w_distance(y_train_loc, y_test_glob):
    return wasserstein_distance(np.argmax(y_train_loc, axis=1), np.argmax(y_test_glob, axis=1))

def make_seed():
    current_time = datetime.now()
    seconds = (current_time - current_time.min).seconds
    rounding = (seconds + 7.5) // 15 * 15
    rounded = current_time + timedelta(seconds=rounding - seconds)
    rounded = rounded.replace(microsecond=0)
    seed = rounded.now().year + rounded.now().month + rounded.now().hour + rounded.now().minute
    #print('Seed:', seed)
    return seed