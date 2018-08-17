'''

Evaluation module for Treepedia 2.0: Applying Deep Learning for Large-scale Quantification of Urban Tree Cover

First appeared on arxiv (https://arxiv.org/abs/1808.04754), submitted and accepted for IEEE BigData Congress 2018

Script to replicate paper results written by Bill Cai

'''

import model_lib as treepedia_dl
import os.path
import numpy as np

print("Loading model from weights_test.hdf5")
model = treepedia_dl.load_keras_mod("weights_test.hdf5")

if not os.path.exists('DOWNLOADED-FILES'):
    raise("Test set images not downloaded! Refer to Github documentation for download instructions")

print("Loading dataset")
imgdata_array,labeldata_array = treepedia_dl.load_and_resize("test_set.txt")
predicted = treepedia_dl.eval_batch(model,imgdata_array,100,1)

diff = np.zeros([len(predicted),1])
for i in range(len(predicted)):
    diff[i] = predicted[i]-labeldata_array[i]
print("Average Absolute GVI error is: %f" % np.mean(np.abs(diff)))
percentile_result= np.percentile(diff,[5,95])
print("5-95 percentile of Absolute GVI error is (%f, %f)" % (percentile_result[0],percentile_result[1]))
print("Correlation of predicted and true GVI is: %f" % np.corrcoef(predicted.T,labeldata_array.T)[0,1])


