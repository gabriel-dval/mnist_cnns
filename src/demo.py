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
    X_train : array
    y_train : array
    X_test : array
    y_test : array
    val_proportion : float

    Returns
    ---
    X_tr : array
    y_tr : array
    X_val : array
    y_val : array
    X_test : array
    y_test : array
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



# train, validation and test function





# Fit function

def fit(epochs, X_train, y_train, X_val, y_val, X_test, y_test, loss_fn, save_loc, early_stopping = True):
    '''Function to load data and fit model for a set number of epochs and
    a set number of protein embeddings (hence the double for loop).

    Arguments
    ---
    epochs : int
        Number of passes through network
    X_train : array
    y_train : array
    X_val : array
    y_val : array
    X_test : array
    y_test : array
    loss_fn : function
        Loss function to be computed
    save_loc : str
        Where should the results be saved ?
    early_stopping : boolean
        Using early stopping ?

    Returns
    ---
    loss_vectors : list of lists
        Lists of loss values for each epoch
    y_preds : list of lists
        TBC
    '''
    start = time.time()

    # Initialise patience
    if early_stopping:
        print("[INFO]: Initializing early stopping")
        early_stopping = EarlyStopping(patience=PATIENCE, min_delta=0)


    # Create test set.
    loss_vector = []
    val_loss_vector = []
    

    test_dataset = CustomIterDataset_Local(df_test, path)
    t_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    
    model = CONV_2Class(embd_dim).to(device)

    # Set optimizer based on model parameters
    lr = LR
    optimizer = optim.Adam(model.parameters(), 
                        lr=lr, amsgrad=False) 

    # Train model for set number of epochs
    for epoch in range(epochs):
        #Create the datasets and dataloaders for training with shuffle each step
        df_learn = df_learn.sample(frac = 1)     #Shuffle of dataframe
        train_data = CustomIterDataset(df_learn, path)
        val_data = CustomIterDataset(df_val, path)
        train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
        val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
        train_loss = train(model, train_loader, loss_fn, optimizer, epoch)
        val_loss = validate(model, val_loader, loss_fn, epoch)
        loss_vector.append(train_loss)
        val_loss_vector.append(val_loss)

        if early_stopping:
            early_stopping(val_loss)
            if early_stopping.early_stop:
                break

    #test_loss, predictions, real_vals, pdbs = test(model, t_loader, loss_fn, epochs, path)
    loss_vectors.append(loss_vector)
    val_loss_vectors.append(val_loss_vector)
    #predictions_across_embds.append(predictions)

    # Save weights of each model
    model_name = f'{path[29:]}_{CURRENT_CV}'
    save_model(epochs, model, optimizer, loss_fn, save_loc, model_name)
    
    #writer.close()
    end = time.time()

    #Save the trained model weights for a final time
    #save_model(epochs, model, optimizer, loss_fn)
    
    print(f"Training time: {(end-start)/60:.3f} minutes\n")
    # print(len(predictions_across_embds))
    print("Ensemble model complete")
    return loss_vectors, val_loss_vectors #, predictions_across_embds, real_vals, pdbs



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

    # 



   

    






