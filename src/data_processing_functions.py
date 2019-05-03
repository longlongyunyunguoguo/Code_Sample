#!/usr/bin/python3

import numpy as np
import pandas as pd
import skimage.io
import os

def img_norm(img):
    '''
    Scale the image to the range [0, 1].
    Input: img: (dim_x, dim_y, channel) int array for an image (if in rgb 
                format)
    Output: (dim_x, dim_y, channel) float array for the same image scaled to 
            [0, 1] range
    '''
    img = (img - img.min()) / (img.max() - img.min())
    return img

def image_ids_in(root_dir):
    '''
    Read in all folder ids from the data directory.
    Input: root_dir: a string for directory to be investigated
    Output: a list of strings for all image ids shown in the directory
    '''
    ids = []
    for ID in os.listdir(root_dir):
        ids.append(ID)
    return ids

def read_image(image_id, pattern, space="rgb"):
    '''
    Read in one image given the directory position.
    Inputs: image_id: a string ID of one image
            pattern: a string for the filepath of the image
            space: a string for the space of the image
    Output: (dim_x, dim_y, channel) int array for an image (if in rgb format)
    '''
    if space not in ["rgb", "hsv"]:
        raise ValueError("Image space has to be rgb or hsv.")
    image_file = pattern.format(image_id, image_id)
    image = skimage.io.imread(image_file)
    
    image = image[:, :, :3] #Drop alpha channel which is not used
    if space == "hsv":
        image = skimage.color.rgb2hsv(image)
    return image

def read_image_labels(image_id, pattern):
    '''
    Read in image segmentation labels by combining each nuclei information.
    Inputs: image_id: a string ID of one image
            pattern: a string for the filepath of the masks for the image
    Output: (dim_x, dim_y) 0-1 int array representing the segmentation of the 
            image (1 as foreground and 0 as background)
    '''
    mask_file = pattern.format(image_id)
    masks = skimage.io.imread_collection(mask_file).concatenate()
    num_masks = masks.shape[0]
    labels = np.zeros((masks.shape[1], masks.shape[2]), np.uint16)
    for index in range(0, num_masks):
        labels[masks[index] > 0] = 1
    return labels

def get_images_details(image_ids, label, img_pattern, mask_pattern):
    '''
    Generate detailed information for all training/test images.
    Inputs: image_ids: a list of strings for all image ids shown in the 
                       directory
            label: a bool variable to indicate whether to extract image 
                   label information
            img_pattern: a string for the filepath of the image
            mask_pattern: a string for the filepath of the masks for the image
    Output: a list of tuples containing detailed image information (string 
            image ids, int arrays for images, 0-1 int arrays for image labels)
    '''
    details = []
    for image_id in image_ids:
        image_rgb = read_image(image_id, img_pattern, space="rgb")
        if label:
            labels = read_image_labels(image_id, mask_pattern)
            info = (image_id, image_rgb, labels)
        else:
            info = (image_id, image_rgb)
        details.append(info)
    return details

def data_process(datatype="train", label=True):
    '''
    Import image information from either training/test data directory for 
    model training/prediction.
    Inputs: datatype: a string to indicate whether the input data is for 
                      training/validation
            label: a bool variable to indicate whether to extract image label 
                   information
    Output: a pandas dataframe containing detailed information for training/
            validation images
    '''
    if datatype not in ["train", "test"]:
        raise ValueError("Data type has to be train or test.")
    if label not in [True, False]:
        raise ValueError("Label information has to be True or False.")
    if datatype == "test" and label == True:
        raise ValueError("Test data does not contain label information.")
    
    root_dir = "../sample_inputs/" + datatype
    img_pattern = "%s/{}/images/{}.png" % root_dir
    mask_pattern = "%s/{}/masks/*.png" % root_dir
    IMAGE_ID = "ImageId"
    IMAGE = "Image"
    LABEL = "ImageLabel"
    
    print("Getting {}ing images ...".format(datatype))
    train_image_ids = image_ids_in(root_dir)
    details = get_images_details(
        train_image_ids, 
        label, 
        img_pattern, 
        mask_pattern,
    )
    if label:
        COLS = [IMAGE_ID, IMAGE, LABEL]
    else:
        COLS = [IMAGE_ID, IMAGE]
        
    return pd.DataFrame(details, columns=COLS)

class DataGenerator(object):
    '''
    Generate image data streams for neural network training/prediction.
    '''
    def __init__(
        self, 
        dim_x=128, 
        dim_y=128, 
        dim_z=3, 
        batch_size=5, 
        shuffle=True,
    ):
        '''
        Initialization.
        Inputs: dim_x: int height of the image fragments feeding into the model
                dim_y: int width of the image fragments feeding into the model
                dim_z: int number of channels of the image fragments feeding 
                       into the model
                batch_size: int number of samples for one batch
                shuffle: bool variable to indicate whether to shuffle samples 
                         for each epoch
        '''
        self.dim_x = dim_x
        self.dim_y = dim_y
        self.dim_z = dim_z
        self.batch_size = batch_size
        self.shuffle = shuffle
    
    def generate(self, list_ids, train_df):
        '''
        Generate data stream for model inputs.
        Inputs: list_ids: a string list of image ids
                train_df: a pandas dataframe containing detailed information 
                          for training images
        Outputs: X (four dimensional float array) and y (four dimensional int 
                   array) for model inputs
        '''
        while True:
            indexes = self.__get_exploration_order(list_ids)
            imax = int(len(indexes) / self.batch_size)
            for i in range(imax):
                list_ids_batch = [
                    list_ids[k] 
                    for k in indexes[i * self.batch_size:min(len(indexes), 
                        (i + 1) * self.batch_size)]
                ]
                X, y = self.__data_generation(list_ids_batch, train_df)
                yield X, y

    def __get_exploration_order(self, list_ids):
        '''
        Generate data sequence for one epoch.
        Input: list_ids: a string list of image ids
        Output: indexes of (shuffled) list_ids
        '''
        indexes = np.arange(len(list_ids))
        if self.shuffle:
            np.random.shuffle(indexes)
        return indexes

    def __data_generation(self, list_ids_batch, train_df):
        '''
        Generate input data for one batch.
        Inputs: list_ids_batch: a string list of image ids for one batch
                train_df: a pandas dataframe containing detailed information 
                          for training images
        Outputs: X (four dimensional float array) and y (four dimensional int 
                   array) for one batch of model inputs
        '''
        X = np.empty((self.batch_size, self.dim_x, self.dim_y, self.dim_z))
        y = np.empty((self.batch_size, self.dim_x, self.dim_y, 1), dtype=int)
      
        for i, ID in enumerate(list_ids_batch):
            whole_img = train_df.loc[train_df["ImageId"] == ID, "Image"].item()
            whole_label = train_df.loc[train_df["ImageId"] == ID, "ImageLabel"].item()
            whole_img = img_norm(whole_img)
            
            #Randomly crop a dim_x * dim_y fragment from the given image
            height, width, _ = whole_img.shape
            starth = np.random.randint(0, height - self.dim_x)
            startw = np.random.randint(0, width - self.dim_y)
            x_img = whole_img[starth:starth + self.dim_x, startw:startw + self.dim_y, :]
            y_label = whole_label[starth:starth + self.dim_x, startw:startw + self.dim_y]
            
            X[i, :, :, :] = x_img
            y[i, :, :, 0] = y_label
        return X, y
    
def extract_frag(img, x, y, crop_x, crop_y):
    '''
    Crop the image with specified size and origin at x, y.
    Inputs: img: a three dimensional float array for the original image
            x: int start point along height for the fragment
            y: int start point along width for the fragment
            crop_x: int height of the fragment
            crop_y: int width of the fragment
    Output: a three dimensional float array for the extracted fragment
    '''
    return img[x : x + crop_x, y : y + crop_y, :]

def addback_frag(img_label, frag, x, y, crop_x, crop_y):
    '''
    Stack fragment values back to the corresponding image label.
    Inputs: img_label: a three dimensional float array for the image segmentation
            frag: a four dimensional float array for the fragment
            x: int start point along height for the fragment
            y: int start point along width for the fragment
            crop_x: int height of the fragment
            crop_y: int width of the fragment
    '''
    img_label[x : x + crop_x, y : y + crop_y, :] += frag[:, :, :]
    
def padding_with_reflection(img):
    '''
    Pad the original image by reflection.
    Input: img: a three dimensional float array for the original image
    Output: a larger three dimensional float array three fold of height & width
            compared with the original image. Padded by reflections.
    '''
    img_tmp = np.atleast_3d(img)
    xlen, ylen, nlayer = img_tmp.shape
    
    img_padded = np.zeros((3 * xlen, 3 * ylen, nlayer))
    img_padded[xlen:2 * xlen, ylen : 2 * ylen, :] = img_tmp
    img_padded[xlen:2 * xlen, 0 : ylen, :] = img_tmp[:, ::-1, :]
    img_padded[xlen:2 * xlen, 2 * ylen : 3 * ylen, :] = img_tmp[:, ::-1, :]
    img_padded[:xlen, :, :] = img_padded[2 * xlen - 1 : xlen - 1 : -1, :, :]
    img_padded[2 * xlen : 3 * xlen, :, :] = img_padded[2 * xlen - 1 : xlen - 1 : -1, :, :]

    return img_padded

def generate_frag_array(
    img, 
    input_x, 
    input_y, 
    output_x, 
    output_y, 
    stride_x, 
    stride_y,
):
    '''
    Generate an array of fragments from a given image.
    Inputs: img: a three dimensional float array for the original image
            input_x: int height of the fragment
            input_y: int width of the fragment
            output_x: int height of the final applied section in the middle of
                      the fragment for segmentation stacking
            output_y: int width of the final applied section in the middle of
                      the fragment for segmentation stacking
            stride_x: int stride size along height to generate fragments
            stride_y: int stride size along width to generate fragments
    Output: a four dimensional float array (#fragments, height, width, #channel)
            of extracted fragments from the original image
    '''
    if output_x % stride_x != 0:
        print("Invalid stride_x. output_x/stride_x must be an integer.")
        return
    if output_y % stride_y != 0:
        print("Invalid stride_y. output_y/stride_y must be an integer.")
        return
    
    img_padded = padding_with_reflection(img=img)
    img = np.atleast_3d(img)
    xlen, ylen, nlayers = img.shape
    start_x = xlen - output_x // 2
    
    estimate = int((xlen + output_x) / stride_x) * int((ylen + output_y) / stride_y) * 2
    tmp = np.zeros((estimate, input_x, input_y, nlayers))
    
    k = 0
    while start_x < xlen * 2:
        start_y = ylen - output_y // 2
        while start_y < ylen * 2:
            tmp[k, :, :, :] = extract_frag(
                img=img_padded,
                x=start_x - int((input_x - output_x) / 2), 
                y=start_y - int((input_y - output_y) / 2), 
                crop_x=input_x,
                crop_y=input_y,
            )
            k += 1
            start_y += stride_y
        start_x += stride_x

    result = np.zeros((k, input_x, input_y, nlayers))
    result[:, :, :, :] = tmp[:k, :, :, :]
    return result
    
def extract_middle_frag(frag, input_x, input_y, output_x, output_y):
    '''
    Extract the middle output_x * output_y section from the given fragments.
    Inputs: frag: a four dimensional float array (#fragments, height, width, 
                 #channel) for the original fragments
            input_x: int height of the fragment
            input_y: int width of the fragment
            output_x: int height of the final applied section in the middle of
                      the fragment for segmentation stacking
            output_y: int width of the final applied section in the middle of
                      the fragment for segmentation stacking
    Output: a four dimensional float array (#sections, height, width, #channel)
            of middle sections from the original fragments
    '''
    nn, _, _, nlayers = frag.shape
    res = np.zeros((nn, output_x, output_y, nlayers))
    res[:, :, :, :] = frag[:, int((input_x - output_x) / 2) : int((input_x + output_x) / 2), 
                         int((input_y - output_y) / 2) : int((input_y + output_y) / 2), :]
    return res

def combine_outputs(img_shape, output, stride_x, stride_y):
    '''
    Add back fragments predicted by the model into final image labels.
    Inputs: img_shape: an int tuple of height and width of one image
            output: a four dimensional float array (#fragments, height, width, 
                 #channel) for the original fragments
            stride_x: int stride size along height when fragments are generated
            stride_y: int stride size along width when fragments are generated
    Output: a three dimensional float array (height, width, #channel) in range 
            [0, 1] for image segmentation prediction
    '''
    xlen, ylen = img_shape
    nn, output_x, output_y, nlayers = output.shape
    img_padded = np.zeros((3 * xlen, 3 * ylen, nlayers))
    start_x = xlen - output_x // 2
    
    k = 0
    while start_x < xlen * 2:
        start_y = ylen - output_y // 2
        while start_y < ylen * 2:
            addback_frag(
                img_label=img_padded, 
                frag=output[k, :, :, :], 
                x=start_x, 
                y=start_y, 
                crop_x=output_x, 
                crop_y=output_y,
            )
            k += 1
            start_y += stride_y
        start_x += stride_x
		
    result = img_padded[xlen : 2 * xlen, ylen : 2 * ylen, :] \
    / ((output_x / stride_x) * (output_y / stride_y))
    return result

def sub_fragment_extract(
    data_df, 
    input_dim=(128, 128), 
    output_dim=(100, 100), 
    stride = (50, 50),
):
    '''
    Crop each image into fragments with input_dim size for model inputs with a 
    stride size of stride.
    For each fragment, only middle output_dim size part is used for final model 
    prediction, since the neural network's accuracy decreases when it comes to 
    the image edges.
    Inputs: data_df: a pandas dataframe containing detailed information 
                     for images
            input_dim: an int tuple containing height & width of fragments
            output_dim: an int tuple containing height & width of middle section
                        of fragments for segmentation predictions
            stride: an int tuple containing stride sizes along height & width
                    when fragments are generated in an image
    Output: a pandas dataframe containing detalied information for fragments
    '''
    input_x, input_y = input_dim
    output_x, output_y = output_dim
    stride_x, stride_y = stride
    details = []
    print("Start to generate fragment dataset ...")
    i = 1
    for index, row in data_df.iterrows():
        img = row["Image"]
        img = img_norm(img)
        img_id = row["ImageId"]
        img_shape = row["Image"].shape[:2]
        X = generate_frag_array(
            img=img, 
            input_x=input_x, 
            input_y=input_y, 
            output_x=output_x, 
            output_y=output_y, 
            stride_x=stride_x, 
            stride_y=stride_y,
        )
        info = (img_id, img_shape, X)
        details.append(info)
        i += 1
        
    COL = ["ImageId", "ImageShape", 'X']
    fragment_data = pd.DataFrame(details, columns=COL)
    return fragment_data
