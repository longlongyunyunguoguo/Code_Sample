#!/usr/bin/python3

from keras.models import Model
from keras.layers import (
    Input, 
    Conv2D, 
    BatchNormalization, 
    Activation,
)
from keras.optimizers import Adam
from keras.layers.merge import concatenate
from keras import backend as K

def jaccard_coef(y_true, y_pred):
    '''
    Calculate Jaccard coefficient between two arrays of image segmentations.
    Inputs: y_true: (batch_size, dim_x, dim_y, 1) 0-1 int array for true image 
                    segmentations
            y_pred: (batch_size, dim_x, dim_y, 1) float array within the range
                    of [0, 1] for predicted image segmentations
    Output: Jaccard coefficient
    '''
    SMOOTH = 1e-12
    intersection = K.sum(y_true * y_pred, axis=[0, 1, 2])
    sum_ = K.sum(y_true + y_pred, axis=[0, 1, 2])
    jac = (intersection + SMOOTH) / (sum_ - intersection + SMOOTH)
    return K.mean(jac)

def jaccard_loss(y_true, y_pred):
    '''
    Calculate 1 - Jaccard coefficient between two arrays of image segmentations.
    Inputs: y_true: (batch_size, dim_x, dim_y, 1) 0-1 int array for true image 
                    segmentations
            y_pred: (batch_size, dim_x, dim_y, 1) float array within the range 
                    of [0, 1] for predicted image segmentations
    Output: 1 - Jaccard coefficient (To be minimized during training)
    '''
    return 1 - jaccard_coef(y_true, y_pred)

def cnn_block(X, filter_num, kernel_size):
    '''
    A set of layers to extract features from previous layer outputs.
    Inputs: X: (batch_size, dim_x, dim_y, channel) float array for image fragments
            filter_num: int number of filters for the convolutional layer
            kernel_size: int size of convolutional kernels
    Output: features generated from the convolutional block
    '''
    X = Conv2D(
        filters=filter_num, 
        kernel_size=(kernel_size, kernel_size), 
        kernel_initializer="he_normal", 
        padding="same")(X)
    X = BatchNormalization()(X)
    X = Activation("relu")(X)
    return X

def model_gen(input_dim, lr=0.001):
    '''
    One example customized convolutional neural network for image segmentation.
    Inputs: input_dim: (dim_x, dim_y) int dimension of the input image fragment
            lr: learning rate
    Output: a neural network for image segmentation
    '''
    print("Building model ...")
    inputs = Input((input_dim[0], input_dim[1], 3))
    
    local1 = cnn_block(inputs, 8, 3)
    local2 = cnn_block(local1, 8, 10)
    local3 = concatenate([local1, local2])
    local4 = cnn_block(local3, 3, 1)
    
    global1 = cnn_block(local3, 16, 15)
    global2 = cnn_block(global1, 16, 15)
    global3 = cnn_block(global2, 16, 15)
    global4 = cnn_block(global3, 32, 15)
    global5 = cnn_block(global4, 3, 1)
    
    combined = concatenate([inputs, local4, global5])
    outputs = Conv2D(
        filters=1,
        kernel_size=(1, 1),
        kernel_initializer="he_normal",
        activation="sigmoid")(combined)
    
    model = Model(inputs=[inputs], outputs=[outputs])
    adam = Adam(lr=lr)
    model.compile(
        loss=jaccard_loss,
        optimizer=adam,
        metrics=["accuracy"],
    )
    return model
