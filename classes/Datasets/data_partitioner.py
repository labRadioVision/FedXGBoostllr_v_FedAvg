import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
import tensorflow as tf
import numpy as np
import pandas as pd
from classes.Datasets.dataset_utils import kl_divergence, w_distance, make_seed
from classes.params import fl_param, simul_param


def split_iid(x, y, args, type='train'):
    if fl_param.ALL_DATA or type == 'valid':
        local_size = len(x) // fl_param.NUM_CLIENTS
    else:
        local_size = args.samples

    for i in range(fl_param.NUM_CLIENTS):
        dir = f'data/client_{i}/{type}/'
        os.makedirs(dir, exist_ok=True)
        
        start_index = i * local_size
        end_index = (i + 1) * local_size
        x_part = x[start_index:end_index]
        y_part = y[start_index:end_index]
        
        print('Client {}: KL Divergence: {:.4f} | Wasserstein Distance: {:.4f} | Samples {}'.format(i, kl_divergence(y_part, y), w_distance(y_part, y), len(y_part)))
        np.save(dir + f'x_{type}.npy', x_part)
        np.save(dir + f'y_{type}.npy', y_part)
    print(f'Saved {type} data')
    

def split_iid_sim(x, y, samples, clients, type='train'):
    if type == 'valid':
        local_size = len(x) // clients
    else:
        local_size = samples

    for i in range(clients):
        dir = f'data/client_{i}/{type}/'
        os.makedirs(dir, exist_ok=True)
        
        start_index = i * local_size
        end_index = (i + 1) * local_size
        x_part = x[start_index:end_index]
        y_part = y[start_index:end_index]
        
        print('Client {} | Samples {}'.format(i, len(y_part)))
        np.save(dir + f'x_{type}.npy', x_part)
        np.save(dir + f'y_{type}.npy', y_part)
    print(f'Saved {type} data')
  

def split_label(x, y, args):
    #min_require_size = samples if samples is not None else 25
    min_require_size = 50    
    
    y_label = np.argmax(y, axis=-1)
    classes = len(np.unique(y_label))
    #print(classes)
    total_samples = len(x)
    
    min_size = 0

    while (min_size < min_require_size and min_size != np.nan):
        idx_batch = [[] for _ in range(fl_param.NUM_CLIENTS)]
        np.random.seed(make_seed())
        for k in range(classes):
            idx_k = np.where(y_label == k)[0]
            np.random.shuffle(idx_k)
            proportions = np.random.dirichlet(np.repeat(args.alpha, fl_param.NUM_CLIENTS))
            # try to make devices have the same number of samples
            proportions_s = np.sort(proportions)[::-1]
            samples_per_device = np.array([len(b) for b in idx_batch])
            sorted_indices = np.argsort(samples_per_device)
            proportions = [0] * len(sorted_indices)  # Create a new array to store the updated values
            for i in range(len(sorted_indices)):
                proportions[sorted_indices[i]] = proportions_s[i]
            ## Balance
            proportions = np.array([p * (len(idx_j) < total_samples / fl_param.NUM_CLIENTS) for p, idx_j in
                                    zip(proportions, idx_batch)])
            proportions = proportions / proportions.sum()
            proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
            idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
        min_size = min([len(idx_batch[j]) for j in range(fl_param.NUM_CLIENTS)])
    print('min_size ', min_size, '\n')

        
    for i in range(fl_param.NUM_CLIENTS):
        dir = f'data/client_{i}/train/'
        os.makedirs(dir, exist_ok=True)
        
        np.random.shuffle(idx_batch[i])

        samples = len(idx_batch[i]) if (fl_param.ALL_DATA or len(idx_batch[i]) <= args.samples) else args.samples
        idxs = np.random.choice(idx_batch[i], size=samples, replace=False)
        
        x_part = tf.gather(x, idxs)
        y_part = tf.gather(y, idxs)
        
        print('Client {}: KL Divergence: {:.4f} | Wasserstein Distance: {:.4f} | Samples {}'.format(i, kl_divergence(y_part, y), w_distance(y_part, y), len(y_part)))
        np.save(dir + 'x_train.npy', x_part)
        np.save(dir + 'y_train.npy', y_part)
    print('Saved train data')


def split_sample(x, y, args):
    #samples_imbalance(device_index, samples, min_require_size, alpha, total_samples):
    total_samples = x.shape[0]
    min_require_size = 50  
    idxs = np.random.permutation(total_samples)
    min_size = 0
    min_prob = np.round(min_require_size / total_samples, 5)
    while min_size < min_require_size:
        np.random.seed(make_seed())
        proportions = np.random.dirichlet(np.repeat(args.alpha, fl_param.NUM_CLIENTS))
        if np.any(proportions < min_prob):
            proportions = np.where(proportions < min_prob, min_require_size / total_samples, proportions)
            # Normalization
            proportions[proportions != min_prob] = np.round(
                proportions[proportions != min_prob] / np.sum(proportions[proportions != min_prob]) / (
                            1 - np.sum(proportions[proportions == min_prob])), 5)
            proportions = np.round(proportions / proportions.sum(), 5)
        else:
            proportions = np.round(proportions / proportions.sum(), 5)
        min_size = np.min(proportions * len(idxs))
    proportions = (np.cumsum(proportions) * len(idxs)).astype(int)[:-1]
    idx_batch = np.split(idxs, proportions)
    

    if fl_param.ALL_DATA:  # limit the max number of samples
        scaling_factor = 1
    else:
        scaling_factor = args.samples / max([len(idx_batch[i]) for i in range(fl_param.NUM_CLIENTS)])
        
    for i in range(fl_param.NUM_CLIENTS):
        dir = f'data/client_{i}/train/'
        os.makedirs(dir, exist_ok=True)
        
        np.random.shuffle(idx_batch[i])

        # make sure at least 'min_require_size' is per device
        samples = int(scaling_factor * len(idx_batch[i])) if int(
            scaling_factor * len(idx_batch[i])) > min_require_size else min_require_size
    
        idxs = np.random.choice(idx_batch[i], size=samples, replace=False)
        
        x_part = tf.gather(x, idxs)
        y_part = tf.gather(y, idxs)
        
        print('Client {}: KL Divergence: {:.4f} | Wasserstein Distance: {:.4f} | Samples {}'.format(i, kl_divergence(y_part, y), w_distance(y_part, y), len(y_part)))
        np.save(dir + 'x_train.npy', x_part)
        np.save(dir + 'y_train.npy', y_part)
    print('Saved train data')
    
    
def split_feature(x, y, args):
    if args.data == 'mnist' or args.data == 'medmnist':
        
        if fl_param.ALL_DATA or type == 'valid':
            local_size = len(x) // fl_param.NUM_CLIENTS
        else:
            local_size = args.samples
        
        for i in range(fl_param.NUM_CLIENTS):
            dir = f'data/client_{i}/{type}/'
            os.makedirs(dir, exist_ok=True)
            
            start_index = i * local_size
            end_index = (i + 1) * local_size
            x_part = x[start_index:end_index]
            y_part = y[start_index:end_index]
            
            x_part_noisy = x_part + args.alpha*i/fl_param.NUM_CLIENTS * np.random.normal(loc=0.0, scale=1.0, size=x_part.shape) 
            print('Client {}: KL Divergence: {:.4f} | Wasserstein Distance: {:.4f}'.format(i, kl_divergence(y_part, y), w_distance(y_part, y)))
            np.save(dir + f'x_{type}.npy', x_part_noisy)
            np.save(dir + f'y_{type}.npy', y_part)
        
    if args.data == 'stroke':
        even_samples = True # True - split into bins with equal num of samples, else split into equal intervals
        split_feature = 'age'  # LEAST CORR -'bmi'; MOST CORR -'age'
        df = x.copy().reset_index(drop=True)
        split_feature_bin = split_feature + '_bin'
        
        if even_samples:  # split the feature st the number of samples per interval is the same
            print('Even split wrt the samples\n')
            df[split_feature_bin] = pd.qcut(df[split_feature], fl_param.NUM_CLIENTS, duplicates='drop')
        else:  # split the feature into even intervals
            print('Even split wrt the feature interval\n')
            df[split_feature_bin] = pd.cut(df[split_feature], bins=fl_param.NUM_CLIENTS)
        
        df[split_feature_bin] = df[split_feature_bin].apply(lambda x: x.mid)

        # Create a list of datasets based on the intervals
        for i, mid_point in enumerate(np.sort(df[split_feature_bin].unique())):
            print('Mid_point{} : {:.2f}'.format(i, mid_point))
            dir = f'data/client_{i}/{type}/'
            os.makedirs(dir, exist_ok=True)
            
            idx_batch = df[df[split_feature_bin] == mid_point].index
            
            samples = len(idx_batch) if (fl_param.ALL_DATA or len(idx_batch) <= args.samples) else args.samples
            idxs = np.random.choice(idx_batch, size=samples, replace=False)
        
            x_part = tf.gather(x, idxs)
            y_part = tf.gather(y, idxs)
            
            
            #x_part = tf.gather(x, idx)
            #y_part = tf.gather(y, idx)
            
            print('Client {}: KL Divergence: {:.4f} | Wasserstein Distance: {:.4f} | Samples {}'.format(i, kl_divergence(y_part, y), w_distance(y_part, y), len(y_part)))
            np.save(dir + f'x_{type}.npy', x_part)
            np.save(dir + f'y_{type}.npy', y_part)
    print('Saved train data')
    
    


