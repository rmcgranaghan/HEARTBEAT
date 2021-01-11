In this directory are two notebooks for creating validation and training training data.  

DMSP_OMNI_Pipeline_1min_5min-_SH_NH_combined-Example.ipynb

is for creating new 1 minute training data from the DMSP and NASAOmni databases. There are some training examples for the total electron flux and the total electron counts as well as the channel counts.  Note that to run this notebook there are two options. The first is to download the pre-run file from: https://drive.google.com/drive/u/1/folders/15OfhoPURNNrRcJVkVwVrxHvUtitJeMMSand then begin training.  The other option is to redownload all the cdf files (~120 GB) for the DMSP 1 second data and then recreate a new training file(s).  

DMSP_OMNI_Pipeline_1-sec-storm_extract_compute-download-version.ipynb

In this example, very similar to the above example, only validation data is downloaded and created for a specified date range and spacecraft id.  This data is 1 second data.
 
