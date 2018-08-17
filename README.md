# Fast and Accurate Estimation of Green View Index
Replication of results for the following paper: [Treepedia 2.0: Deep Learning Based Large Scale Quantification of Urban Canopy Cover](https://arxiv.org/abs/1808.04754), which is currently published as an arxiv preprint, and has been accepted/will appear in the IEEE BigData Congress 2018 Conference Proceedings.

This project is by the Treepedia team at the [MIT Senseable City Lab](http://senseable.mit.edu). It is also meant to be an extension of the [existing Treepedia repository](https://github.com/mittrees/Treepedia_Public) by directly replacing the GreenView_Calculate.py script. Further details are provided below.

We would love to hear how this has helped in your own projects (contact us at [senseable-trees@mit.edu]{mailto:senseable-trees@mit.edu}), and would appreciate (but not require) citation and attribution if you have used our code/models in any of your projects.

## Requirements and setup

We recommend using Python 3.5. There have been documented problems of users who load weights in Python 3.6 from models trained in Python 3.5 (see [issue](https://github.com/keras-team/keras/issues/9595)).

Packages that are used include keras, tensorflow (or tensorflow_gpu), PIL and scikit-image. You can install them using pip:
```
pip install tensorflow keras Pillow scikit-image
```

After doing so, you can clone this repository and also download the trained model weights and Google Street View (GSV) images. Copyright belongs to Google for original images, ground truth images are licensed under the MIT license, copyright belongs to the original authors of the paper - Bill Cai, Xiaojiang Li, Ian Seiferling and Carlo Ratti.

```
git clone git@github.com:billcai/treepedia_dl_public.git
```

Google Drive link for trained model weights: Download here
Google Drive link for GSV images: Download here

*you may use gdrive download file_id if you have installed gdrive*

Next, unzip the files and you are ready!

## Replicating results

Simply run 

```
python rep_results.py
```

This script runs the direct end-to-end GVI model, where a ResNet (residual network) was trained to directly estimate the Green View Index. A sample output of running the above command is:

```
Average Absolute GVI error is: 0.047427
5-95 percentile of Absolute GVI error is (-0.098191, 0.083860)
Correlation of predicted and true GVI is: 0.939691
```

## Replacing GreenView_Calculate.py

We replaced the *VegetationClassification* function in the old GreenView_Calculate.py with *predict_single* function. Results in the paper show that we should expect increase in evaluation speed and Green View Index (GVI) accuracy. 

If you wish to incorporate *predict_single* into your own script, *predict_single* has the following arguments:
```python
def predict_single(model, image, verbosity)
```

**model** is a loaded Keras model, which can be loaded from a hdf5 saved file using *keras.models.load_model*. **image** is a numpy array that must be in the format of [H,W,3]. **verbosity** determines what is printed; verbosity=1 means that evaluation time estimates would be printed as well.


