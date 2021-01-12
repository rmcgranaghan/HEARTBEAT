In this directory are two notebooks for creating validation and training training data.  

DMSP_OMNI_Pipeline_1min_5min-_SH_NH_combined-Example.ipynb

is for creating new training data from the DMSP and NASAOmni databases. There are some training examples for the total electron flux and the total electron counts as well as the channel counts.  Note that to run this notebook there are two options. The first is to download the pre-run file from: and then begin training.  The other option is to redownload all the cdf files for the DMSP 1 second data and then recreate a new training file(s).  

DMSP_OMNI_Pipeline_1-sec-storm_extract_compute-download-version.ipynb

In this example, very similar to the above example, only validation data is downloaded and created for a specified date range and spacecraft id.  This data is 1 second data.

In the subdirectory final_models_Jan_2021

There is a notebook that trains models with a tail loss function for the total_flux and the total counts as well as the channel counts.  Most of these models are for the case of 59 (rather than the full 148) inputs that correspond only to geomagnetic data and not the solar wind data.  This examples are also saved with their scaling functions which are in pickle files, so that they can be used to evaluating without loading the whole training database to get the scaling 
