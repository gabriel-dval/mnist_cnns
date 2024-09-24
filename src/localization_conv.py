'''This script runs a convolution model on a new data, the localization data set
'''
#Import modules

import os
import re
import time
import argparse
import random
import datetime as dt
import requests
import zipfile
from pandas import read_csv

import numpy as np
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
    #plt.savefig(f"{figure_path}/{index}_BS{BATCH_SIZE}_LR{LR}.png")
    plt.show()



# Custom Dataset --------------------------------------------------------------------------

def download_data():
    url="https://github.com/JJAlmagro/subcellular_localization/raw/refs/heads/master/notebook%20tutorial/data/reduced_train.npz"
    datasetFolderPath = "data/localization/"
    file = "reduced_train.npz"
    FilePath = os.path.join(datasetFolderPath, file)

    if not os.path.exists(datasetFolderPath):
        os.makedirs(datasetFolderPath)

    def download_file(url, filename):
        response = requests.get(url, stream=True)
        with tqdm.wrapattr(open(filename, "wb"), "write", miniters=1,
                           total=int(response.headers.get('content-length', 0)),
                           desc=filename) as fout:
            for chunk in response.iter_content(chunk_size=4096):
                fout.write(chunk)

    # Download the zip file if it does not exist
    if not os.path.exists(FilePath):
        download_file(url, FilePath)


def train_validation_test(X_train, y_train, mask_train, val_proportion = 0.1, test_proportion = 0.2):
    '''Simple function to create our train, validation and test sets
    Based on previously handed out format of data.
    
    Args
    ---
    X_train : array
    y_train : array
    mask_train : array
    val_proportion : float
    test_proportion : float


    Returns
    ---
    X_tr : array
    y_tr : array
    X_val : array
    y_val : array
    X_test : array
    y_test : array
    '''
    # Shuffle data set
    ns =  X_train.shape[0]
    shuffle_index = np.random.permutation(ns)
    train_images, train_labels, masks = X_train[shuffle_index,:,:], y_train[shuffle_index], mask_train[shuffle_index]


    # Set test and validation index
    val_size = int(ns * val_proportion)
    test_size = int(ns * test_proportion)

    val_index = val_size + test_size

    # Create train, test and validation
    X_tr = train_images[:-val_index,:,:]
    y_tr = train_labels[:-val_index]
    masks_tr = masks[:-val_index]
    X_val = train_images[-val_index:-test_size,:,:]
    y_val = train_labels[-val_index:-test_size]
    masks_val = masks[-val_index:-test_size]
    X_te = train_images[-test_size:,:,:]
    y_te = train_labels[-test_size:]
    masks_te = masks[-test_size:]
    
    return X_tr, y_tr, masks_tr, X_val, y_val, masks_val, X_te, y_te, masks_te


def prep_data(X, y, mask):
    '''Function to prepare localization dataset

    Args
    ---
    X : array
        Features 
    y : array
        Associated label
    mask : array
        Provided vector for masking parts of the image
    '''
    # Masking of the image
    masked_image = np.transpose(X) * mask
    masked_image = np.transpose(masked_image)
    tensor_X = torch.from_numpy(masked_image.astype(np.float32))

    # Turn labels into class vectors
    tensor_y = torch.as_tensor(y, dtype = torch.long)
    oh_y = nn.functional.one_hot(tensor_y, num_classes = 10)
    oh_y = oh_y.to(torch.float32)
    
    # Yield images
    return tensor_X, oh_y

class LocalizationCustom(Dataset):
    '''Custom dataset for processing the Localization data'''
    def __init__(self, X_path, y_path, mask):
        self.X_path = X_path  # X
        self.y_path = y_path  # Y
        self.mask = mask      # Masking vector
    
    def __len__(self):
        return len(self.y_path)
    
    def __getitem__(self, idx):
        image = self.X_path[idx, :, :]
        label = self.y_path[idx]
        mask = self.mask[idx]
        return prep_data(image, label, mask)


# Cross-validation generation --------------------------------------------------------------

def cross_val_data(X_train, y_train, cross_val_nb):
    '''Function that generates the train, validation and test data for
    the number of folds specified.

    Args
    ---
    X_train : array
        Training image data.
    y_train : array
        Training labels.
    X_test : array
        Test image data.
    y_test : array
        Test labels.
    cross_val_nb : int
        Number of cross-validation folds.

    Returns
    ---
    folds : list of tuples
        Each tuple contains (X_train_fold, y_train_fold, X_val_fold, y_val_fold)
        for each cross-validation fold.
    '''
    # Shuffle the entire dataset
    ns = X_train.shape[0]
    shuffle_index = np.random.permutation(ns)
    images, labels = X_train[shuffle_index], y_train[shuffle_index]

    # Determine the number of samples in each fold
    fold_size = ns // cross_val_nb
    
    # Cross-validation data
    folds = []

    for i in range(cross_val_nb):
        # Define validation set for the current fold
        start_val = i * fold_size
        end_val = (i + 1) * fold_size if i != cross_val_nb - 1 else ns
        
        X_val_fold = images[start_val:end_val]
        y_val_fold = labels[start_val:end_val]
        
        # Define training set for the current fold (exclude the validation fold)
        X_train_fold = np.concatenate((images[:start_val], images[end_val:]), axis=0)
        y_train_fold = np.concatenate((labels[:start_val], labels[end_val:]), axis=0)
        
        # Append the fold to the list as a tuple (X_train_fold, y_train_fold, X_val_fold, y_val_fold)
        folds.append((X_train_fold, y_train_fold, X_val_fold, y_val_fold))
    
    return folds


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
        # plt.figure(figsize = (10, 6))
        # sns.heatmap(overall_conf_matrix, annot = True)  
        # plt.title(f'CM - Epochs : {EPOCHS} ; Batch size : {BATCH_SIZE}; Learning Rate : {LR}')
        # plt.savefig(f"{figure_path}/CONVConfusionMatrix_BS{BATCH_SIZE}_LR{LR}.png")


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


def fit(epochs, X_train, y_train, mask_train, X_val, y_val, mask_val, X_test, y_test, mask_test, 
        loss_fn, save_loc, early_stopping = True):
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
    

    test_dataset = LocalizationCustom(X_test, y_test, mask_test)
    t_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    
    model = CONV().to(device)

    # Set optimizer based on model parameters
    lr = LR
    optimizer = optim.Adam(model.parameters(), 
                        lr=lr ,amsgrad = False) 

    # Train model for set number of epochs
    for epoch in range(epochs):
        #Create the datasets and dataloaders for training with shuffle each step
        train_data = LocalizationCustom(X_train, y_train, mask_train)
        val_data = LocalizationCustom(X_val, y_val, mask_val)
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
        self.conv1 = nn.Conv1d(20, 32, kernel_size=3, padding=0)
        self.conv2 = nn.Conv1d(32, 16, kernel_size=3, padding=0)
        self.pool1 = nn.MaxPool1d(kernel_size=2)
        self.lin1 = nn.Linear(1568, 10)
        
        self.batchnorm1 = nn.BatchNorm1d(32)
        self.batchnorm2 = nn.BatchNorm1d(16)
        self.flatten = nn.Flatten()
        
        self.dropout1 = nn.Dropout(0.3)

        self.relu = nn.ReLU()
        self.softmax = torch.nn.Softmax(dim = 1)

    def forward(self, x):

        out = x.permute(0, 2, 1)  # [batch, length, features] --> [batch, features, length]
        
        out = self.conv1(out)
        out = self.dropout1(out)
        out = self.batchnorm1(out)
        out = self.relu(out)

        #out = self.pool1(out)
        out = self.conv2(out)
        out = self.batchnorm2(out)
        out = self.relu(out)

        out = self.pool1(out)
        out = self.flatten(out)
        out = self.lin1(out)
        out = self.softmax(out)

        return out


class Block(nn.Module):
    def __init__(self, filters, subsample=False):
        super().__init__()
        """
        2-layer residual learning building block
        
        Args
        --- 
        filters:   int
                     the number of filters for all layers in this block
                   
        subsample: boolean
                     whether to subsample the input feature maps with stride 2
                     and doubling in number of filters
                     
        Attributes
        ---
        shortcuts: boolean
                     When false the residual shortcut is removed
                     resulting in a 'plain' convolutional block.
        """
        # Subsampling
        s = 0.5 if subsample else 1.0
        
        # Setup layers
        self.conv1 = nn.Conv1d(int(filters*s), filters, kernel_size=3, 
                               stride=int(1/s), padding=1, bias=False)
        self.bn1   = nn.BatchNorm1d(filters, track_running_stats=True)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv1d(filters, filters, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2   = nn.BatchNorm1d(filters, track_running_stats=True)
        self.relu2 = nn.ReLU()

        # Shortcut downsampling
        self.downsample = nn.MaxPool1d(kernel_size=1, stride=2) 
        
    def shortcut(self, z, x):
        """ 
        Shortcut option
        
        Args
        ---
        x: tensor
             the input to the block
        z: tensor
             activations of block prior to final non-linearity
        """
        if x.shape != z.shape:
            d = self.downsample(x)
            p = torch.mul(d, 0)
            return z + torch.cat((d, p), dim=1)
        else:
            return z + x        
    
    def forward(self, x, shortcuts=False):

        z = self.conv1(x)
        z = self.bn1(z)
        z = self.relu1(z)
        
        z = self.conv2(z)
        z = self.bn2(z)
        
        # Shortcut connection
        if shortcuts:
            z = self.shortcut(z, x)

        z = self.relu2(z)
        
        return z
    

class ResNet(nn.Module):
    def __init__(self, n, shortcuts=True):
        super().__init__()
        self.shortcuts = shortcuts
        
        # Input
        self.convIn = nn.Conv1d(20, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.dropout1 = nn.Dropout(0.3)
        self.bnIn   = nn.BatchNorm1d(16, track_running_stats=True)
        self.relu   = nn.ReLU()

        # Stack2
        self.stack1 = Block(32, subsample=False)

        # Stack3
        self.stack2 = Block(64, subsample=True)
        
        # Output
        self.avgpool = nn.AvgPool2d((1, 1))
        self.fcOut   = nn.Linear(6400, 10, bias=True)
        self.softmax = nn.LogSoftmax(dim=-1)     
        
        
    def forward(self, x):  

        z = x.permute(0, 2, 1)  # [batch, length, features] --> [batch, features, length]

        z = self.convIn(z)
        z = self.dropout1(z)
        z = self.bnIn(z)
        z = self.relu(z)
        
        # Residual blocks
        z = self.stack1(z, shortcuts=self.shortcuts)
        
        z = self.stack2(z, shortcuts=self.shortcuts)

        z = self.avgpool(z)
        z = z.view(z.size(0), -1)
        z = self.fcOut(z)
        z = self.softmax(z)

        return z
    



if __name__ == '__main__':
    
    # Set seed
    set_seed(42)

    # Quick test of the data
    image_path = 'data/localization/reduced_train.npz'

    dataset = np.load(image_path)
    X_train = dataset["X_train"]
    y = dataset["y_train"]
    mask_train = dataset["mask_train"]

    # Computation device

    device = ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Computation device: {device}\n")

    # Set hyperparameters
    PATIENCE = 5
    BATCH_SIZE = 128
    NUM_WORKERS = 0
    EPOCHS = 60
    LR = 0.0005
    LOSS_FN = nn.CrossEntropyLoss(reduction = 'none')

    tx, ty, tm, vx, vy, vm, tex, tey, tem = train_validation_test(X_train, y, mask_train)

    loss_vector, val_loss_vector = fit(EPOCHS, tx, ty, tm, vx, vy, vm, tex, tey, tem, 
                                       LOSS_FN, None, early_stopping = True)
    
    plot_loss_function(None, None, loss_vector, val_loss_vector)


    # Fit model
    # losses = []
    # val_losses = []
    # for k in folds:
    #     tx, ty, vx, vy = k
    #     loss_vector, val_loss_vector = fit(EPOCHS, tx, ty, vx, vy, tex, tey, LOSS_FN, None, early_stopping = True)
    #     losses.append(loss_vector)
    #     val_losses.append(val_loss_vector)

    # plt.figure(figsize = (10, 6))
    # for i, (l, val) in enumerate(zip(losses, val_losses)):
    #     colour = np.random.rand(3,)
    #     plt.plot(list(range(len(loss_vector))), loss_vector, color = colour, label = f'CV{i+1} training loss')
    #     plt.plot(list(range(len(val_loss_vector))), val_loss_vector, color = colour, label = f'CV{i+1} validation loss',
    #             linestyle = 'dashed')
    # plt.xlabel("Number of epochs")
    # plt.ylabel("Loss value")
    # plt.title(f'Loss function - Epochs : {EPOCHS} ; Batch size : {BATCH_SIZE}; Learning Rate : {LR}')
    # plt.legend(loc = 'upper right')
    # plt.savefig(f"../results/CONVCVLosses_BS{BATCH_SIZE}_LR{LR}.png")


    # Plots - will plot loss function, confusion matrix and maybe ROC
    #plot_loss_function('../results', 'CONVLoss', loss_vector, val_loss_vector)
    




   

    










