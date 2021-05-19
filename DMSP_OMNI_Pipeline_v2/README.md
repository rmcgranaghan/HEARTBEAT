README.md
In this directory are notebooks for creating validation and training training data.

This directory is setup similarily to the original DMSP_OMNI_Pipeline, with the main difference being the numberflux calculation has been fixed to include the geometric factor

the March 17th 2013 storm is validated with these models. Many plots, just for spacecraft id 17 are created. This plots are at the second level which is higher resolution than the minute level data that was used to train this models. There are three different models used, one for othe flux, one for othe counts, and one for the channel counts. A CSV file for all the results is created for all spacecrafts, 16,17,18 , and the one second data inputs that were used to evaluate the models are also saved.
