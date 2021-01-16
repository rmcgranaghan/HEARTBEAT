There is a notebook that trains models with a tail loss function for the total_flux and the total counts as well as the channel counts. Most of these models are for the case of 59 (rather than the full 148) inputs that correspond only to geomagnetic data and not the solar wind data. This examples are also saved with their scaling functions which are in pickle files, so that they can be used to evaluating without loading the whole training database to get the scaling

Also in this directory are the two newer files: new file: DMSP_OMNI_Pipeline_final_models_Jan-no_minute_Averages.ipynb new file: March_17_storm_final_models_Jan-no_minute_Averages.ipynb

In the first file, new models are trained with 33 inputs. This inputs only have averages at the hour level and not the minute level. Also these models use custom loss function for the tails of the distributions

In the second file the March 17th 2013 storm is validated with these models. Many plots, just for spacecraft id 17 are created. This plots are at the second level which is higher resolution than the minute level data that was used to train this models. There are three different models used, one for othe flux, one for othe counts, and one for the channel counts. A CSV file for all the results is created for all spacecrafts, 16,17,18 , and the one second data inputs that were used to evaluate the models are also saved.
