# MNIST and Localization dataset classification

A short project aiming to train and test multiple CNN architectures on the commonly used Handwritten MNIST images and another dataset

## Getting Started

These instructions will describe the contents of this repo and help navigate the available files. There are 3 python scripts, each encoding a different model (except the localization script, which implements both a convolutional model
and a ResNet). These are only to to track the changes and development of the models - please view the PetitProjet_DUVAL.ipynb notebook for the detailed steps of data analysis and model architectures.

### Prerequisite packages

To run the following scripts and notebook, a few different modules are necessary (see the scripts to see necessary imports). An environment .yml file (mnist.yml) has been provided to if needed. This program was written using Python v3.10.

### Installing

If you wish to use the provided environment, you can install it with conda 
using this command.

macOS command line

    conda env create -f mnist.yml

Or you can install all packages directly
    
    conda create -n mnist numpy pandas pytorch tqdm matplotlib seaborn requests scikit-learn scipy



## Running the scripts/notebooks

Each script runs the training for the models; they can be run using the following bash command

macOS command line

    python mnist_fc.py

When running the notebook or the scripts, please make sure you input the proper data path and keep the images associated
with the notebook in its directory. 


## Authors

  - **GabrielDuval** - *Provided README Template* -
    [gabriel-dval](https://github.com/gabriel-dval)