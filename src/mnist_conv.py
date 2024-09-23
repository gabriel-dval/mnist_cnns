'''Similar model as previous, but using convolutions
instead of flattening the images.
'''
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
from torch.utils.data import IterableDataset, DataLoader, Dataset

from tqdm.autonotebook import tqdm

from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay, balanced_accuracy_score
from sklearn.utils.class_weight import compute_class_weight, compute_sample_weight
from sklearn.metrics import roc_curve, RocCurveDisplay, PrecisionRecallDisplay, matthews_corrcoef

import matplotlib.pyplot as plt
import seaborn as sns


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


# Plotting functions

def plot_loss_function(figure_path, index, loss_vector, val_loss_vector):
    '''Basic function to plot trajectory of loss

    Arguments
    ---
    loss_vector : list
        Values of loss through iterations of model

    Return
    ---
    Nothing - plots function
    '''
    time = dt.date.today()

    plt.figure(figsize = (10, 6))
    plt.plot(list(range(len(loss_vector))), loss_vector, color = 'red', label = 'Training Loss')
    plt.plot(list(range(len(val_loss_vector))), val_loss_vector, color = 'blue', label = 'Validation loss',
             linestyle = 'dashed')
    plt.xlabel("Number of epochs")
    plt.ylabel("Loss value")
    plt.title(f'Loss function - Epochs : {EPOCHS} ; Batch size : {BATCH_SIZE}; Learning Rate : {LR}')
    plt.legend(loc = 'upper right')
    plt.savefig(f"{figure_path}/{index}_BS{BATCH_SIZE}_LR{LR}.png")



# Custom Dataset --------------------------------------------------------------------------

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
    # Shuffle train data set and test data set
    ns =  X_train.shape[0]
    shuffle_index = np.random.permutation(ns)
    train_images, train_labels = X_train[shuffle_index,:,:,:], y_train[shuffle_index]

    ns =  X_test.shape[0]
    shuffle_index = np.random.permutation(ns)
    test_images, test_labels = X_test[shuffle_index,:,:,:], y_test[shuffle_index]

    # Set validation index
    val_size = int(ns * val_proportion)

    # Create train, test and validaiton
    X_tr = train_images[:-val_size,:,:,:]
    y_tr = train_labels[:-val_size]
    X_val = train_images[-val_size:,:,:,:]
    y_val = train_labels[-val_size:]
    X_te = test_images
    y_te = test_labels
    
    return X_tr, y_tr, X_val, y_val, X_te, y_te


def prep_2d_image(X, y):
    '''Function to prepare image dataset

    Args
    ---
    X : array
        Single 2D image 
    y : array
        Associated label
    '''
    # No transformation of the image
    tensor_X = torch.from_numpy(X.astype(np.float32))

    # Turn labels into class vectors
    tensor_y = torch.as_tensor(y, dtype = torch.long)
    oh_y = nn.functional.one_hot(tensor_y, num_classes = 10)
    oh_y = oh_y.to(torch.float32)
    
    # Yield images
    return tensor_X, oh_y

class MNISTCustom(Dataset):
    '''Custom dataset for processing the MNIST images'''
    def __init__(self, X_path, y_path):
        self.X_path = X_path  # X
        self.y_path = y_path  # Y
    
    def __len__(self):
        return len(self.y_path)
    
    def __getitem__(self, idx):
        image = self.X_path[idx, :, :, :]
        label = self.y_path[idx]
        return prep_2d_image(image, label)


# train, validation and test function ------------------------------------------------------


def train(model, train_loader, loss_fn, optimizer, epoch):
    '''Train function for the model. To evaluate the training, multiple different
    measures are used: Balanced Accuracy, Precision, Recall.

    Arguments
    ---
    model : nn.Module descendant
        Model through which training data is passed
    train_loader : DataLoader
        Training data with features and labels
    loss_fn : function
        Method of loss calculation
    optimizer : function (taken using torch.optim)
        Optimisation method
    epoch : int
        Number of passes through the network

    Returns
    ---
    epoch_loss : ?
        Information on loss (loss vector ?)
    '''
    model.train()

    train_loss = 0
    counter = 0

    confusion_matrices = []

    all_true_classes = []
    all_pred_classes = []

    labels = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

    with tqdm(train_loader, total=len(train_loader), unit="batch") as tepoch:
        for X, y in tepoch:
            counter += 1
            tepoch.set_description(f"Epoch {epoch}")

            #Send the input to the device
            X, y = X.to(device), y.to(device)

            #Compute prediction and loss
            pred = model(X)

            # Loss calculation
            loss = torch.sum(loss_fn(pred, y))

            #Backpropagation
            optimizer.zero_grad(set_to_none = True)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
        
            # Batch metrics - first detach vectors
            pred = pred.detach().cpu().numpy()
            y = y.detach().cpu().numpy()

            class_preds = np.argmax(pred, axis = 1)
            true = np.argmax(y, axis = 1)

            # Save class information     
            all_true_classes.extend(true)
            all_pred_classes.extend(class_preds)
           
            #Progres pbar
            postfix = {}
            postfix["Train: loss"] = f"{train_loss / counter:.5f}"
            tepoch.set_postfix(postfix)
        
        #Loss and protein metrics
        epoch_loss = train_loss / counter

        # overall_conf_matrix = confusion_matrix(all_true_classes, all_pred_classes)
        # overall_performance = classification_report(all_true_classes, all_pred_classes, 
        #                                             target_names = labels, zero_division=0.0)
        ba = balanced_accuracy_score(all_true_classes, all_pred_classes)
        mc = matthews_corrcoef(all_true_classes, all_pred_classes)


        #Prints
        print(f"\n Training performance across images : \n")
        print(f"Balanced Accuracy Score : {ba}")
        print(f"Matthews CC : {mc}")
    
    return epoch_loss


def validate(model, val_loader, loss_fn, epoch):
    '''Train function for the model. 

    Arguments
    ---
    model : nn.Module descendant
        Model through which training data is passed
    val_loader : DataLoader
        Validation data with features and labels
    loss_fn : function
        Method of loss calculation
    optimizer : function (taken using torch.optim)
        Optimisation method
    epoch : int
        Number of passes through the network

    Returns
    ---
    epoch_loss : ?
        Information on loss (loss vector ?)
    '''
    model.eval()

    val_loss = 0
    counter = 0

    confusion_matrices = []

    all_true_classes = []
    all_pred_classes = []

    labels = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

    with torch.no_grad():
        for X, y in val_loader:
            counter += 1

            #Send the input to the device
            X, y = X.to(device), y.to(device)

            #Compute prediction and loss
            pred = model(X)

            # Loss calculation
            loss = torch.sum(loss_fn(pred, y))
            val_loss += loss.item()
        
            # Batch metrics - first detach vectors
            pred = pred.detach().cpu().numpy()
            y = y.detach().cpu().numpy()

            class_preds = np.argmax(pred, axis = 1)
            true = np.argmax(y, axis = 1)

            # Save class information     
            all_true_classes.extend(true)
            all_pred_classes.extend(class_preds)
        
        #Loss and protein metrics
        epoch_loss = val_loss / counter

        # overall_conf_matrix = confusion_matrix(all_true_classes, all_pred_classes)
        # overall_performance = classification_report(all_true_classes, all_pred_classes, 
        #                                             target_names = labels, zero_division=0.0)
        ba = balanced_accuracy_score(all_true_classes, all_pred_classes)
        mc = matthews_corrcoef(all_true_classes, all_pred_classes)
    
        

        #Prints
        print(f"\n Validation performance across images : \n")
        print(f"Balanced Accuracy Score : {ba}")
        print(f"Matthews CC : {mc}")
        
    return epoch_loss


def test(model, test_loader, loss_fn, epoch, figure_path):
    '''Train function for the model. 

    Arguments
    ---
    model : nn.Module descendant
        Model through which training data is passed
    val_loader : DataLoader
        Validation data with features and labels
    loss_fn : function
        Method of loss calculation
    optimizer : function (taken using torch.optim)
        Optimisation method
    epoch : int
        Number of passes through the network

    Returns
    ---
    epoch_loss : ?
        Information on loss (loss vector ?)
    '''
    model.eval()

    test_loss = 0
    counter = 0

    all_true_classes = []
    all_pred_classes = []

    labels = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

    with torch.no_grad():
        for X, y in test_loader:
            counter += 1

            #Send the input to the device
            X, y = X.to(device), y.to(device)

            #Compute prediction and loss
            pred = model(X)

            # Loss calculation
            loss = torch.sum(loss_fn(pred, y))
            test_loss += loss.item()
        
            # Batch metrics - first detach vectors
            pred = pred.detach().cpu().numpy()
            y = y.detach().cpu().numpy()

            class_preds = np.argmax(pred, axis = 1)
            true = np.argmax(y, axis = 1)

            # Save class information     
            all_true_classes.extend(true)
            all_pred_classes.extend(class_preds)
        
        #Loss and protein metrics
        epoch_loss = test_loss / counter

        overall_conf_matrix = confusion_matrix(all_true_classes, all_pred_classes, normalize='true')
        # overall_performance = classification_report(all_true_classes, all_pred_classes, 
        #                                             target_names = labels, zero_division=0.0)
        ba = balanced_accuracy_score(all_true_classes, all_pred_classes)
        mc = matthews_corrcoef(all_true_classes, all_pred_classes)

        # Draw and save confusion matrix
        plt.figure(figsize = (10, 6))
        sns.heatmap(overall_conf_matrix, annot = True)  
        plt.title(f'CM - Epochs : {EPOCHS} ; Batch size : {BATCH_SIZE}; Learning Rate : {LR}')
        plt.savefig(f"{figure_path}/CONVConfusionMatrix_BS{BATCH_SIZE}_LR{LR}.png")


        #Prints
        print(f"\n Test performance across images : \n")
        print(f"Balanced Accuracy Score : {ba}")
        print(f"Matthews CC : {mc}")
    
    return epoch_loss, all_pred_classes, all_true_classes


# Fit function -----------------------------------------------------------------------------

class EarlyStopping():
    """
    Early stopping to stop the training when the loss does not improve after
    certain epochs.
    """
    def __init__(self, patience=10, min_delta=0):
        """
        :param patience: how many epochs to wait before stopping when loss is
               not improving
        :param min_delta: minimum difference between new loss and old loss for
               new loss to be considered as an improvement
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss == None:
            self.best_loss = val_loss
        elif self.best_loss - val_loss > self.min_delta:
            self.best_loss = val_loss
            # reset counter if validation loss improves
            self.counter = 0
        elif self.best_loss - val_loss < self.min_delta:
            self.counter += 1
            print(f"[INFO]: Early stopping counter - {self.counter} of {self.patience}")
            if self.counter >= self.patience:
                print('[INFO]: Early stopping')
                self.early_stop = True


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
    

    test_dataset = MNISTCustom(X_test, y_test)
    t_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    
    model = CONV().to(device)

    # Set optimizer based on model parameters
    lr = LR
    optimizer = optim.RMSprop(model.parameters(), 
                        lr=lr) 

    # Train model for set number of epochs
    for epoch in range(epochs):
        #Create the datasets and dataloaders for training with shuffle each step
        train_data = MNISTCustom(X_train, y_train)
        val_data = MNISTCustom(X_val, y_val)
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

    test_loss, predictions, real_vals = test(model, t_loader, loss_fn, epochs, '../results')

    # Save weights of each model
    # model_name = f'{path[29:]}_{CURRENT_CV}'
    # save_model(epochs, model, optimizer, loss_fn, save_loc, model_name)
    
    end = time.time()
    
    print(f"Training time: {(end-start)/60:.3f} minutes\n")
    # print(len(predictions_across_embds))
    print("Model complete")
    return loss_vector, val_loss_vector #, predictions_across_embds, real_vals, pdbs


# Models ------------------------------------------------------------------------------------

class CONV(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=0)
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.lin1 = nn.Linear(5408, 10)
        
        self.batchnorm1 = nn.BatchNorm1d(16)
        self.batchnorm2 = nn.BatchNorm1d(16)
        self.flatten = nn.Flatten()
        
        self.dropout1 = nn.Dropout(0.3)

        self.relu = nn.ReLU()
        self.softmax = torch.nn.Softmax(dim = 1)

    def forward(self, x):

        out = x.permute(0, 3, 1, 2)  # [batch, width, height, channels] --> [batch, channels, width, height]
        
        out = self.conv1(out)
        out = self.relu(out)
        out = self.pool1(out)
        #out = self.batchnorm1(out)
        #out = self.dropout1(out)
        # out = self.conv2(out)
        # out = self.batchnorm2(out)
        # out = self.relu(out)
        out = self.flatten(out)
        out = self.lin1(out)
        out = self.softmax(out)

        return out



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

    # unique, counts = np.unique(tey, return_counts=True)
    # plt.bar(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'], counts)
    # plt.show()

    # Computation device

    device = ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Computation device: {device}\n")

    # Set hyperparameters
    PATIENCE = 5
    BATCH_SIZE = 128
    NUM_WORKERS = 0
    EPOCHS = 60
    LR = 0.001
    LOSS_FN = nn.CrossEntropyLoss(reduction = 'none')

    # Fit model
    loss_vector, val_loss_vector = fit(EPOCHS, tx, ty, vx, vy, tex, tey, LOSS_FN, None, early_stopping = True)


    # Plots - will plot loss function, confusion matrix and maybe ROC
    plot_loss_function('../results', 'CONVLoss', loss_vector, val_loss_vector)
    




   

    






