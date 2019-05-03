## Code Sample Description

The code is to solve an image segmentation problem using a customized neural network. Image segmentation is applied to identify all nuclei in medical images from different fluorescent methods. Since images are not necessarily of same size, multiple same sized fragments were generated for each image as model inputs, and their predictions were stacked back to original image space to have final predictions.

## Folder Content

### ./codes

This folder contains all python3 codes for the sample project. One can implement the codes by running main.py.

constants.py: Contains all constants used for the sample codes

data_processing_functions.py: Contains all relevant functions for data processing

learning_functions.py: Contains all relevant functions for training/predicting with a given neural network model

main.py: Main function to train and evaluate a neural network with small sample datasets

ZoomNet_jac_0402.py:		One example customized neural network

### ./model_outputs

This folder contains all output files generated from main.py

### ./sample_inputs:		

This folder contains sample images/labels. Data from https://www.kaggle.com/c/data-science-bowl-2018
