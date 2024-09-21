'''Test script to load data and train CNN models

All models will be implemented using pytorch
'''

# Necessary modules

#Import modules

import os
import re
import time
import argparse
import random
import datetime as dt

import numpy as np
#import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from torch.utils.data import IterableDataset, DataLoader

from tqdm.autonotebook import tqdm

from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay, balanced_accuracy_score
from sklearn.utils.class_weight import compute_class_weight, compute_sample_weight
from sklearn.metrics import roc_curve, RocCurveDisplay, PrecisionRecallDisplay

import matplotlib.pyplot as plt


# Seed function stolen from Yann --------------------------------------------------------

def set_seed(seed = 42):
    """Fix the seeds for reproductible runs during training"""
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set as {seed}")


# Visualise dataset -----------------------------------------------------------------------

def view_images(X_path, y_path):
    '''Function to view 10 images provided using matplotlib

    Args
    ---
    X_path : str
        Path to images
    y_path : str
        Path to labels

    Returns
    ---
    Nothing
    '''
    # Load images
    images = np.load(X_path)
    labels = np.load(y_path)

    # Display them
    plt.figure(figsize=[10,10])
    for i in range (9):    # for first 9 images
        plt.subplot(3, 3, i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(images[i], cmap=plt.cm.binary)
        plt.xlabel(labels[i])

    plt.show()


# Custom Dataset 

def train_validation_test(X_train, y_train, X_test, y_test, val_proportion = 0.125):
    '''Simple function to create our train, validation and test sets
    Based on previously handed out format of data.
    
    Args
    ---
    X_train : 
    y_train :
    X_test : 
    y_test : 
    val_proportion

    Returns
    ---
    X_tr :
    y_tr :
    X_val :
    y_val :
    X_test : 
    y_test : 
    '''
    # Shuffle train data set
    ns =  X_train.shape[0]
    shuffle_index = np.random.permutation(ns)
    train_images, train_labels = X_train[shuffle_index,:,:,:], y_train[shuffle_index]

    # Set validation index
    val_size = int(ns * val_proportion)

    # Create train, test and validaiton
    X_tr = train_images[:-val_size,:,:,:]
    y_tr = train_labels[:-val_size]
    X_val = train_images[-val_size:,:,:,:]
    y_val = train_labels[-val_size:]
    
    return X_tr, y_tr, X_val, y_val, X_test, y_test



def prep_flat_image(X, y):
    '''Function to prepare image dataset

    Args
    ---
    X : iterable
        List or array of 2D images 
    y : iterable
        List or array of equivalent labels
    '''
    # Little sanity check
    if len(X) != len(y):
        raise ValueError('Features and labels do not have the same length')
    
    # Flatten images and turn into tensors
    pixels = X.shape[1] * X.shape[2] #784
    flat_X = X.reshape(X.shape[0], pixels)
    tensor_X = torch.from_numpy(flat_X.astype(np.float32))

    # Turn labels into class vectors
    num_classes = np.unique(y_train).shape[0]
    tensor_y = torch.as_tensor(y, dtype = torch.long)
    oh_y = nn.functional.one_hot(tensor_y, num_classes = num_classes)
    
    # Yield images
    for image, label in zip(tensor_X, oh_y):
        yield image, label



class CustomIterDataset(IterableDataset):
    '''Class to create custom iterable dataset from protein embedding
    '''
    def __init__(self, dataframe, X_path, y_path):
        super(CustomIterDataset, self).__init__()
        # `dataframe` is a dataframe with PDB IDs and other PDB stats
        self.dataframe = dataframe
        self.X_path = X_path  # X
        self.y_path = y_path  # Y

    def __len__(self):
        return len(self.dataframe)

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
                return prep_flat_image(self.dataframe, self.X_path, self.y_path)
        else: # in a worker process
            # split workload
            worker_id = worker_info.id
            worker_total_num = worker_info.num_workers
            
            sub_df = np.array_split(self.dataframe, worker_total_num)

            #Add multiworker functionality and sampling from all replicates option
            return prep_flat_image(sub_df[worker_id], self.X_path, self.y_path)



# Train, Validation and Test function




# Fit function



if __name__ == '__main__':
    
    # Set seed
    set_seed(42)

    # Quick test of the images
    image_path = 'data'

    X_train = np.load('../data/train_images.npy')
    y_train = np.load('../data/train_labels.npy')
    X_test = np.load('../data/test_images.npy')
    y_test = np.load('../data/test_labels.npy')

    tx, ty, vx, vy, tex, tey = train_validation_test(X_train, y_train, X_test, y_test)

    print(tx.shape)
    print(ty.shape)
    print(vx.shape)
    print(vy.shape)

   

    






