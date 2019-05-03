#!/usr/bin/python3

import data_processing_functions as dpf
from keras.callbacks import ModelCheckpoint, History
import pandas as pd
import numpy as np

def model_fitting(
    ids, 
    total_ids, 
    cv_id, 
    train_df, 
    model, 
    track_name, 
    input_dim=(128, 128), 
    epoch_num=10, 
    batch_size=5
):
    '''
    Train a neural network model with specified training/validation images for 
    one cross-fold.
    Inputs: ids: a tuple containing list of image indexes for training/validation
                for one cross-fold
            total_ids: a string list of image ids
            cv_id: an int variable to indicate the number of current cross-fold
            train_df: a pandas dataframe containing detalied information for 
                      images
            model: a Keras neural network model
            track_name: a string containing the name information for the model
            input_dim: an int tuple containing height & width of fragments for
                       model inputs
            epoch_num: number of epochs for training
            batch_size: number of samples for each batch
    '''
    train_ids = [total_ids[i] for i in ids[0]]
    val_ids = [total_ids[i] for i in ids[1]]
	
    filepath = "../model_outputs/{}_{:d}.hdf5".format(track_name, cv_id)
    mcp_save = ModelCheckpoint(
        filepath, 
        save_best_only=True, 
        monitor="val_loss", 
        mode="min"
    )
    history = History()
    params = {
        "dim_x": input_dim[0],
        "dim_y": input_dim[1],
        "dim_z": 3,
        "batch_size": batch_size,
        "shuffle": True,
    }
    training_generator = dpf.DataGenerator(**params).generate(
        train_ids, train_df)
    validation_generator = dpf.DataGenerator(**params).generate(
        val_ids, train_df)
    
    print("{:d}th CV is being performed ...".format(cv_id))
    output_history = model.fit_generator(
        generator=training_generator, 
        steps_per_epoch=max(1, len(train_ids) // batch_size), 
        epochs = epoch_num, validation_data = validation_generator, 
        validation_steps = max(1, len(val_ids) // batch_size), 
        callbacks = [mcp_save, history]
    )
    df = pd.DataFrame.from_dict(history.history)
    filepath = "../model_outputs/History_{}_{:d}.csv".format(track_name, cv_id)
    df.to_csv(
        filepath, 
        sep="\t", 
        index=True, 
        float_format="%.3f"
    )
    del output_history

def fragment_predict(cv_id, model, track_name, image_data):
    '''
    Predict image segmentation for each fragment of the input images using 
    model from one cross-fold.
    Inputs: cv_id: an int variable to indicate the number of current cross-fold
            model: a Keras neural network model
            track_name: a string containing the name information for the model
            image_data: a pandas dataframe containing detalied information for 
                        images for segmentation prediction
    Output: a list of int arrays for image segmentation prediction
    '''
    print("{:d}th cv model used for prediction ...".format(cv_id))
    filepath = "../model_outputs/{}_{:d}.hdf5".format(track_name, cv_id)
    model.load_weights(filepath=filepath)
    
    pred_label = []
    for t in range(image_data.shape[0]):
        print("{:d}th image is being processed ... ({:d}/{:d})".format(
            t + 1, 
            t + 1, 
            image_data.shape[0]))
        x = image_data.loc[t, 'X']
        y_pred = model.predict(x)
        pred_label.append(y_pred) 
    del model
    return pred_label

def model_predict(
    cv_num, 
    data_df, 
    model, 
    track_name, 
    input_dim, 
    output_dim, 
    stride, 
    cutoff,
):
    '''
    Predict image segmentation using averaged outputs from n models (n is 
    number of cross-fold).
    Each model is the best one with lowest validation loss from the 
    corresponding cross-fold training.
    Inputs: cv_num: int number of cross-folds
            data_df: a pandas dataframe containing detalied information for 
                     images for segmentation prediction
            model: a Keras neural network model
            track_name: a string containing the name information for the model
            input_dim: an int tuple containing height & width of fragments
            output_dim: an int tuple containing height & width of middle section
                        of fragments for segmentation predictions
            stride: an int tuple containing stride sizes along height & width
                    when fragments are generated in an image
            cutoff: a float variable to indicate the threshold to recognize a 
                    pixel to be foreground/background (if above cutoff then 
                    foreground, otherwise background)
    Output: a list of tuples (string id, 0-1 int three dimensional array (height, 
            width, 1) for segmentation prediction for each image)
    '''
    img_data = dpf.sub_fragment_extract(
        data_df, 
        input_dim=input_dim, 
        output_dim = output_dim, 
        stride = stride,
    )
    
    print("Start to predict ...")
    for i in range(1, cv_num + 1):
        if i == 1:
            pred_outputs_kfold = np.array(
                fragment_predict(i, model, track_name, img_data))
        else:
            pred_outputs_kfold += np.array(
                fragment_predict(i, model, track_name, img_data))

    pred_outputs_kfold = pred_outputs_kfold / cv_num
    
    print("Start to organzie predictions using average outputs ...")
    image_pred_label = []
    for i in range(img_data.shape[0]):
        pred_test = pred_outputs_kfold[i]
        pred_label = dpf.extract_middle_frag(
            pred_test, 
            input_dim[0], 
            input_dim[1], 
            output_dim[0], 
            output_dim[1],
        )
        output_img = dpf.combine_outputs(
            img_shape=img_data.loc[i, "ImageShape"], 
            output=pred_label, 
            stride_x=stride[0], 
            stride_y=stride[1]
        )
        output_img = np.where(output_img > cutoff, 1, 0)
        image_pred_label.append((img_data.loc[i, "ImageId"], output_img))
    return image_pred_label