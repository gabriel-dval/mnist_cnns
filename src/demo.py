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
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from torch.utils.data import IterableDataset, DataLoader

from tqdm.autonotebook import tqdm
#from torch.utils.tensorboard import SummaryWriter

from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay, balanced_accuracy_score
from sklearn.utils.class_weight import compute_class_weight, compute_sample_weight
from sklearn.metrics import roc_curve, RocCurveDisplay, PrecisionRecallDisplay



from scipy.stats import spearmanr, pearsonr

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



