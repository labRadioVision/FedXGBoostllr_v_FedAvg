import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

import numpy as np
import warnings
from classes.Datasets.data_loader import load_mnist, load_stroke, load_medmnist, load_stroke_nprep
from classes.Datasets.data_partitioner import split_iid, split_label, split_sample, split_feature

import argparse
import json

warnings.filterwarnings("ignore", category=RuntimeWarning)

parser = argparse.ArgumentParser()
parser.add_argument('-data',  choices=['mnist', 'stroke', 'stroke_noprep'], default='stroke_noprep', help = 'Type of data', type=str)
parser.add_argument("-niid_type", choices=['iid', 'label', 'sample', 'feature'], default='iid', help="Heterogeneity type", type=str)
parser.add_argument('-alpha', default=0.5, help=" alpha for non-iid (sigma for noise)", type=float) # small alpha for non-IID
parser.add_argument('-samples', default=100, help="sets a fixed number samples per device", type=int)
args = parser.parse_args() 

data = args.data
niid_type = args.niid_type
alpha = args.alpha

if __name__ == "__main__":
    
    # load the dataset
    if data == 'mnist':
        print('Loading MNIST')
        x_train, y_train, x_valid, y_valid = load_mnist()
    elif data == 'stroke':
        print('Loading Stroke')
        x_train, y_train, x_valid, y_valid = load_stroke()
    elif data == 'stroke_noprep':
        print('Loading Stroke')
        x_train, y_train, x_valid, y_valid = load_stroke_nprep()
    elif data == 'medmnist':
        print('Loading MedMNIST')
        x_train, y_train, x_valid, y_valid = load_medmnist()
    
    # split the training dataset        
    if niid_type == 'iid':
        print('Splitting IID')
        split_iid(x_train, y_train, args, type='train')
    elif niid_type == 'label':
        print('Splitting Label imbalance')
        split_label(x_train, y_train, args)
    elif niid_type =='sample':
        print('Splitting Sample imbalance')
        split_sample(x_train, y_train, args)
    elif niid_type =='feature':
        print('Splitting Feature imbalance')
        split_feature(x_train, y_train, args)
        
    # split the validation dataset
    split_iid(x_valid, y_valid, args, type='valid')
    
    # save data info to json for PS
    n_classes = np.unique(y_valid, axis=0).shape[0] if np.unique(y_valid, axis=0).shape[0]>2 else 1
    data_info = {
        'input_shape': x_train.shape[1:],
        'num_classes': n_classes, #np.unique(y_valid, axis=0).shape[0],
        'data': args.data,
        'niid_type': args.niid_type,
        'alpha': args.alpha
    }
    dir = "data/server/"
    os.makedirs(dir, exist_ok=True)
    with open(dir + "data_info.json", "w") as outfile:
        json.dump(data_info, outfile)