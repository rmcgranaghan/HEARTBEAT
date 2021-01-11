In this directory are two notebooks for creating validation and training training data.  

DMSP_OMNI_Pipeline_1min_5min-_SH_NH_combined-Example.ipynb

is for creating new 1 minute training data from the DMSP and NASAOmni databases. There are some training examples for the total electron flux and the total electron counts as well as the channel counts.  Note that to run this notebook there are two options. The first is to download the pre-run files from: https://drive.google.com/drive/u/1/folders/15OfhoPURNNrRcJVkVwVrxHvUtitJeMMSand then begin training. These are organized by sc_id and are about 10.4 GB total. The other option is to redownload all the cdf files (~120 GB) for the DMSP 1 second data and then recreate a new training file(s).  Note that the NASA Omni data is only gathered at 5min intervals and therefore has repeats accross the DMSP spacecraft traces which are used at a 1min cadence.   

DMSP_OMNI_Pipeline_1-sec-storm_extract_compute-download-version.ipynb

In this example, very similar to the above example, only validation data is downloaded and created for a specified date range and spacecraft id.  This data is 1 second data. An examplel pretrained model is tested here.  Not that the model in this case used a different database for training.
 
