# -*- coding: utf-8 -*-
"""
Created on Fri Mar  9 14:02:21 2018

Makes a quick DEM validation histogram with a csv text file exported from arcgis 
point shapefile with GPS Z, and DEM Z (added from "add surface information")

@author: jlogan
"""

import pandas as pd
import numpy as np
import seaborn as sns

def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())

#Set input file path here
valfile = '2017-1101-DEM-BackpackTopo-Validation.txt'

#import csv file
valdf = pd.read_csv(valfile)

#calc difference (DEM - GPS)
#change column names here if needed
valdf['diff'] = valdf.DEM_Z - valdf.GPS_Z 

#set seaborn style
sns.set_style('darkgrid')

h =sns.distplot(valdf.diff, bins=30, kde=False, hist_kws=dict(edgecolor="b", linewidth=0.5))
h.set_xlabel('Elevation difference, DEM - GPS [m]')
h.set_ylabel('count [n]')

calcrmse = rmse(valdf.DEM_Z,valdf.GPS_Z)
calcmean = np.mean(valdf.diff)

print('============================\n')
print('n = ' + str(len(valdf)) + '\n')
print('Vertical RMSE = ' + str(calcrmse) + '\n')
print('Mean vertical error = ' + str(calcmean) + '\n')
