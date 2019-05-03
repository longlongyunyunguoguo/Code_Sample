#!/usr/bin/python3

'''
Description: The code is to solve an image segmentation problem using a customized 
neural network. Image segmentation is applied to identify all nuclei in medical images
from different fluorescent methods. Since images are not necessarily of same size, 
multiple same sized fragments were generated for each image as model inputs, and their
predictions were stacked back to original image space to have final predictions.
'''

import ZoomNet_jac_0402 as nn_model
import learning_functions as lf
import data_processing_functions as dpf
from constants import (
    CV_NUM,
    SEED,
    CUTOFF,
    INPUT_DIM,
    OUTPUT_DIM,
    STRIDE,
)
from sklearn.model_selection import KFold
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import numpy as np


track_name = "ZoomNet_jac_0402_{:d}fold".format(CV_NUM)

#Read in training data
train_df = dpf.data_process(datatype="train", label=True)
#Extract train data imageids
total_ids = list(train_df["ImageId"].values)

#Split images into cross-fold sets (fragments from one common image always belong to the same train/val set)
kf = KFold(n_splits=CV_NUM, shuffle=True, random_state=SEED)
ids = list(kf.split(total_ids))

#Start model training
model = nn_model.model_gen(INPUT_DIM, lr=0.001)
for i in range(CV_NUM):
    lf.model_fitting(
        ids[i], 
        total_ids, 
        i + 1, 
        train_df, 
        model, 
        track_name, 
        INPUT_DIM, 
        epoch_num=5, 
        batch_size=3,
    )


#Start to predict image segmentations for training data
train_pred = lf.model_predict(
    CV_NUM, 
    train_df, 
    model, 
    track_name, 
    INPUT_DIM, 
    OUTPUT_DIM, 
    STRIDE, 
    CUTOFF,
)

n_row = 5
n_col = 9
with PdfPages("../model_outputs/training_data_prediction.pdf") as pdf:
    fig, ax = plt.subplots(n_row, n_col, figsize=(20, 15))
    for i in range(n_row):
        for j in range(n_col):
            k = 3 * i + j // 3
            if j % 3 == 0: img_to_show = train_df.loc[k, "Image"]
            elif j % 3 == 1: img_to_show = train_df.loc[k, "ImageLabel"]
            else: img_to_show = np.squeeze(train_pred[k][1], axis=2)
            ax[i][j].imshow(img_to_show)
            ax[i][j].set_xticks([])
            ax[i][j].set_yticks([])
            if i == 0:
                if j % 3 == 0: title = "Nuclei Image"
                elif j % 3 == 1: title = "True Segmentation"
                else: title = "Pred Segmentation"
                ax[i][j].set_title(title)
    pdf.savefig()

#Start to predict image segmentation for test data
test_df = dpf.data_process(datatype="test", label=False)
test_pred = lf.model_predict(
    CV_NUM, 
    test_df, 
    model, 
    track_name, 
    INPUT_DIM, 
    OUTPUT_DIM, 
    STRIDE, 
    CUTOFF,
)

n_row = 5
n_col = 2
with PdfPages("../model_outputs/test_data_prediction.pdf") as pdf:
    fig, ax = plt.subplots(n_row, n_col, figsize=(10, 20))
    for i in range(n_row):
        for j in range(n_col):
            if j % 2 == 0: img_to_show = test_df.loc[i, "Image"]
            else: img_to_show = np.squeeze(test_pred[i][1], axis=2)
            ax[i][j].imshow(img_to_show)
            ax[i][j].set_xticks([])
            ax[i][j].set_yticks([])
            if i == 0:
                if j % 2 == 0: title = "Nuclei Image"
                else: title = "Pred Segmentation"
                ax[i][j].set_title(title)
    pdf.savefig()
