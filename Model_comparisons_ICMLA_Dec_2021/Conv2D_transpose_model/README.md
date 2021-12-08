In this model, a different approach is used which creates the whole 2D Mlat and Mlocaltime map with one evaluation of the model.  
Mlat and Mlocaltime are no longer inputs for evaluation, which means that a fixed resolution of 128x128 is used for the 45 to 90 degrees magnetic latitude

A special loss function is used to train this model to produce a 2D output using Conv2D_transpose layersIn this directory are notebooks for creating validation and training training data.
3
​
4
This directory is setup similarily to the or
