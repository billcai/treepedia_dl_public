'''

Evaluation module for Treepedia 2.0: Applying Deep Learning for Large-scale Quantification of Urban Tree Cover

First appeared on arxiv (https://arxiv.org/abs/1808.04754), submitted and accepted for IEEE BigData Congress 2018

Module written by Bill Cai

'''

import keras
from keras.models import load_model
import numpy as np
from PIL import Image
from skimage.transform import resize

current_model = [224,224,3]

def load_keras_mod(model_loc):
    return(load_model(model_loc))

# Warning - this is not memory efficient, in-place resizing would save much more memory
def load_and_resize(image_loc):
    test_data = []
    test_gt = []
    with open(image_loc,'r') as training:
        content = training.readlines()
    for line in content:
        paths = line.split()
        if len(paths) == 2:
            test_data.append(paths[0])
            test_gt.append(paths[1].replace("\n",""))
        if len(paths) == 3:
            test_data.append(paths[0] + " "+paths[1])
            test_gt.append(paths[2].replace("\n",""))
    imgdata = []
    for path1 in test_data:
        imgdata.append((Image.open(path1)).resize(current_model[0:2]))
    labeldata = []
    for path in test_gt:
        labeldata.append(np.sum(np.asarray((Image.open(path)).resize(current_model[0:2]))!=0)*1.0/
                        (current_model[0]*current_model[1]))
    imgdata_array = np.zeros([len(imgdata),current_model[0],current_model[1],current_model[2]])
    labeldata_array = np.zeros([len(imgdata),1])
    for i in range(len(imgdata)):
        imgdata_array[i,:,:,:] = imgdata[i]
        labeldata_array[i,:] = labeldata[i]
    del(imgdata,labeldata)
    return(imgdata_array,labeldata_array)

def eval_batch(model,imgdata_array,batch_size,verbosity=0):
    return(model.predict(x=imgdata_array,batch_size=batch_size,verbose=verbosity))

def predict_single(model,image,verbosity=0):
    if len(np.shape(image)) != 3 or np.shape(image)[2] != 3:
        raise("Expected image to be of [H,W,3]")
    if len(np.shape(image)) != current_model:
        image = resize(image, current_model)
    return(model.predict(x=image,batch_size=1,verbose=verbosity))
