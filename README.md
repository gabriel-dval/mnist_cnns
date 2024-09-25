# mnist_cnns

A short project aiming to train and test multiple CNN architectures on the commonly used Handwritten MNIST images and another dataset

## Getting Started

These instructions will give you a copy of the project up and running on
your local machine for development and testing purposes.

### Prerequisite packages

To run the following scripts, a few different modules are necessary (see the scripts to see necessary imports). An environment .yml file (mnist.yml) has been provided to if needed. This program was written using Python v3.12.6

### Installing

If you wish to use the provided environment, you can install it with conda 
using this command.

macOS command line

    conda env create -f remc_env.yml

You can equally create a new environment with the three packages if this is 
simpler.

    conda create -n myenv python=3.12.5 numpy=2.1 matplotlib=3.9.2 tqdm=4.66.5


## Running the program

The program is run from the command line like any python script. Below is an 
example with a sequence of length 20 called 'test'. Although most parameters have
default options, please make sure to input an optimal energy even if it is not 
known. A cutoff time for each run can equally be passed as argument.

Example usage

    python remc.py HPHPPHHPHPPHPHPH test -5

IMPORTANT - make sure the components.py file is in the same directory as remc.py
so all necessary modules can be imported. 

The program outputs a results_log.txt file, containing results for each iteration 
of the simulation, and a FIGURES directory with the final conformation for each run.

### Launch an example proteins

Launch a simulation of protein S1-1

    python remc.py HPHPPHHPHPPHPHHPPHPH S1 -9


## Authors

  - **GabrielDuval** - *Provided README Template* -
    [gabriel-dval](https://github.com/gabriel-dval)