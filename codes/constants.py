#!/usr/bin/python3

#Number of cross folds and random seed
CV_NUM = 3
SEED = 932894

#Confidence threshold for nuclei identification, any pixel with model output
#higher than cutoff value is predicted to as foreground, if not then background
CUTOFF = 0.5

###Fragment parameters
#For training, 128 * 128 fragments are cropped from each image.
#For prediction, the image is croped into 128 * 128 fragments with specified strides (100 * 100),
#and only middle 100 * 100 part for each prediction is pasted together for a final prediction 
INPUT_DIM = (128, 128)
OUTPUT_DIM = (100, 100)
STRIDE = (100, 100)