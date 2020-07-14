#!/usr/bin/env python
# coding: utf-8

import numpy as np
import os

os.system('source /home/jackalak/heartbeat_work/cdf38_0-dist/bin/definitions.B')
os.environ["CDF_LIB"] = '/home/jackalak/heartbeat_work/cdf38_0-dist/lib'

import pandas as pd
import seaborn as sns
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()
import datetime
from os.path import isfile, join
from sys import getsizeof
import glob

from random import *
from sklearn import *
from scipy import interpolate
from ovationpyme import ovation_utilities
from ovationpyme.ovation_utilities import robinson_auroral_conductance
from ovationpyme.ovation_utilities import brekke_moen_solar_conductance
import geospacepy
from geospacepy import special_datetime,astrodynamics2,satplottools
import aacgmv2 #available on pip



import numpy as np
import matplotlib.pyplot as plt
from ovationpyme.ovation_prime import FluxEstimator,AverageEnergyEstimator,BinCorrector, SeasonalFluxEstimator
from ovationpyme.ovation_utilities import calc_avg_solarwind
from ovationpyme.ovation_plotting import latlt2polar,polar2dial,pcolor_flux
import ipywidgets
from collections import OrderedDict
from functools import partial

# Note ions are disabled for now
auroral_types = ['diff','mono','wave']#,'ions']
flux_types = [
                "energy",
                "number",
                "average energy"
            ]

import datetime
from collections import OrderedDict

import numpy as np
import functools

from geospacepy import omnireader, special_datetime, astrodynamics2

_ovation_prime_omni_cadence = 'hourly' #Ovation Prime was created using hourly SW




atype =auroral_types[0] #widgets['atype_select'].value
jtype = 'energy'#widgets['jtype_select'].value
bincorrect = True#widgets['bincorrect_toggle'].value
combine_hemispheres = True#widgets['combineNS_toggle'].value
manual_newell = -1#widgets['newell_float'].value


file_load_df_cumulative = 'ML_DB_subsamp_ext_full_dfCumulative_complexHemisphereCombine.csv'
DMSP_DATA_DIR=''
df_cumulative = pd.read_csv(os.path.join(DMSP_DATA_DIR,file_load_df_cumulative))
df_cumulative = df_cumulative.sort_values(by=[ 'Datetimes'])
df_cumulative = df_cumulative.set_index('Datetimes')
df_cumulative.index = pd.to_datetime(df_cumulative.index)

cols_to_drop_validation = [c for c in df_cumulative.columns if ('STD' in c) | ('AVG' in c)]#] | ('SC_AACGM_LTIME'==c)]

df_cumulative = df_cumulative.drop(columns=cols_to_drop_validation)


# Separate training and testing data
mask_val = [(df_cumulative.index.year == 2010) & (df_cumulative['ID_SC'].values==16)]
df_val = df_cumulative[mask_val[0]].copy(deep=True)
df_train = df_cumulative.copy(deep=True).drop( df_cumulative.index[mask_val[0]])
print('validation data shape = {}'.format(df_val.shape))
print('train data shape = {}'.format(df_train.shape))


ovation_input = [df_val.index,df_val['SC_AACGM_LAT'],df_val['SC_AACGM_LTIME']]


no_cdf_list = []

times = ovation_input[0]
lats = ovation_input[1]
longs = ovation_input[2]
ovation_fluxes_val = np.zeros((4,len(ovation_input[0])))

seasons = ['spring','summer','fall','winter']



for atype_index in range(len(auroral_types)):


    print(atype_index)
    dt = times[-1]
    jd = ovation_utilities.check_jd(dt)  

    atype =auroral_types[atype_index] #widgets['atype_select'].value  
    estimator = FluxEstimator(atype,jtype)
    seasonal_flux_estimators = {season:SeasonalFluxEstimator(season,atype,jtype) for season in seasons}
    print('starting ', atype_index)
    for i in range(0,len(times)):

        dt = times[i]
        lat = lats[i]
        long = longs[i]

        if lat >= 50.:

            lat_index = int(round((lat - 50.)/.50632911))
            long_index = int(round(long/ 0.25263158))     

            jd_last = jd.copy()
            try:
                #print(dt)
                jd = ovation_utilities.check_jd(dt)
            except Exception as e: 
                print(str(e))
                print(dt)
                no_cdf_list.append(dt)

            if jd[0] != jd_last[0]: 

                doy = dt.timetuple().tm_yday
                weights = estimator.season_weights(doy)
                try:
                    dF = ovation_utilities.calc_dF(dt)
                except Exception as e: 
                    #print(str(e))
                    #print(dt)
                    no_cdf_list.append(dt)


            seasonfluxesN,seasonfluxesS = {},{}
            for season,s_estimator in seasonal_flux_estimators.items():
                gridfluxN,gridfluxS = s_estimator.get_gridded_flux_i_j(dF,long_index, lat_index)
                seasonfluxesN[season]=gridfluxN
                seasonfluxesS[season]=gridfluxS

            gridflux=0
            #print(weights.items())
            for season,W in weights.items():
                gridfluxN,gridfluxS = seasonfluxesN[season],seasonfluxesS[season]
                if combine_hemispheres:
                    gridflux += W*(gridfluxN+gridfluxS)/2
            ovation_fluxes_val[atype_index,i] = gridflux


            #print(i, ovation_fluxes[i], dt, lat,long,lat_index, long_index)

        else:
            #print(i, lat, long)
            ovation_fluxes_val[atype_index,i] =0

print('done')            
import pickle           
with open('ovation_fluxes_val','wb') as f: pickle.dump(ovation_fluxes_val, f)
#with open('ovation_fluxes_val','rb') as f: ovation_fluxes_val = pickle.load(f)



# Now for the training data

ovation_input = [df_train.index,df_train['SC_AACGM_LAT'],df_train['SC_AACGM_LTIME']]


no_cdf_list = []

times = ovation_input[0]
lats = ovation_input[1]
longs = ovation_input[2]
ovation_fluxes_train = np.zeros((4,len(ovation_input[0])))

seasons = ['spring','summer','fall','winter']



for atype_index in range(len(auroral_types)):


    print(atype_index)
    dt = times[-1]
    jd = ovation_utilities.check_jd(dt)  

    atype =auroral_types[atype_index] #widgets['atype_select'].value  
    estimator = FluxEstimator(atype,jtype)
    seasonal_flux_estimators = {season:SeasonalFluxEstimator(season,atype,jtype) for season in seasons}
    print('starting ', atype_index)
    for i in range(0,len(times)):

        dt = times[i]
        lat = lats[i]
        long = longs[i]

        if lat >= 50.:

            lat_index = int(round((lat - 50.)/.50632911))
            long_index = int(round(long/ 0.25263158))     

            jd_last = jd.copy()
            try:
                #print(dt)
                jd = ovation_utilities.check_jd(dt)
            except Exception as e: 
                print(str(e))
                print(dt)
                no_cdf_list.append(dt)

            if jd[0] != jd_last[0]: 

                doy = dt.timetuple().tm_yday
                weights = estimator.season_weights(doy)
                try:
                    dF = ovation_utilities.calc_dF(dt)
                except Exception as e: 
                    #print(str(e))
                    #print(dt)
                    no_cdf_list.append(dt)

            seasonfluxesN,seasonfluxesS = {},{}
            for season,s_estimator in seasonal_flux_estimators.items():
                gridfluxN,gridfluxS = s_estimator.get_gridded_flux_i_j(dF,long_index, lat_index)
                seasonfluxesN[season]=gridfluxN
                seasonfluxesS[season]=gridfluxS

            gridflux=0
            #print(weights.items())
            for season,W in weights.items():
                gridfluxN,gridfluxS = seasonfluxesN[season],seasonfluxesS[season]
                if combine_hemispheres:
                    gridflux += W*(gridfluxN+gridfluxS)/2
            ovation_fluxes_train[atype_index,i] = gridflux

            #print(i, ovation_fluxes[i], dt, lat,long,lat_index, long_index)

        else:
            #print(i, lat, long)
            ovation_fluxes_train[atype_index,i] =0

print('done')            
import pickle           
with open('ovation_fluxes_train','wb') as f: pickle.dump(ovation_fluxes_train, f)
#with open('ovation_fluxes_train','rb') as f: ovation_fluxes_train = pickle.load(f)
