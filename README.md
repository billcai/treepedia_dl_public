# Fast and Accurate Estimation of Green View Index
Replication of results for the following paper: [Treepedia 2.0: Deep Learning Based Large Scale Quantification of Urban Canopy Cover](https://arxiv.org/abs/1808.04754), which is currently published as an arxiv preprint, and has been accepted/will appear in the IEEE BigData Congress 2018 Conference Proceedings.

This project is by the Treepedia team at the [MIT Senseable City Lab](http://senseable.mit.edu). It is also meant to be an extension of the [existing Treepedia repository](https://github.com/mittrees/Treepedia_Public) by directly replacing the GreenView_Calculate.py script. Further details are provided below.

## Requirements and setup

TBC

## Replicating results

TBC

## Replacing GreenView_Calculate.py

We replaced the *VegetationClassification* function in the old GreenView_Calculate.py with *predict_single* function. Results in the paper show that we should expect increase in evaluation speed and Green View Index (GVI) accuracy. 

If you wish to incorporate *predict_single* into your own script, *predict_single* has the following arguments:
```python
def predict_single(model, image, verbosity)
```

**model** is a loaded Keras model, which can be loaded from a hdf5 saved file using *keras.models.load_model*. **image** is a numpy array that must be in the format of [H,W,3]. **verbosity** determines what is printed; verbosity=1 means that evaluation time estimates would be printed as well.


