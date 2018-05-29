# -*- coding: utf-8 -*-
"""
Created on Sat May 19 22:21:13 2018

Create a class for DEM and DEM validation.  Prefer functions in demValidate.py

@author: jlogan
"""

import rasterio
import numpy as np
import pandas as pd
from scipy import ndimage

class dem(object):
    def __init__(self, demfile):
        with rasterio.open(demfile) as src:
            self.dataset = src
            self.bounds = self.dataset.bounds
            self.nodatavals = self.dataset.nodatavals
            self.crs = self.dataset.crs
            
            self.array = self.dataset.read(1)
            self.array[self.array == self.dataset.nodatavals] = np.nan
                
        
    def validate(self, checkfile):
        # get affine transform
        a = self.dataset.affine
        
        # load check points into dataframe
        df = pd.read_csv(checkfile)
        # rename z column to distinguish from dem
        df.rename(columns={'z': 'gps_z'}, inplace=True)
        
        # use affine to get DEM row, column into df
        df['demcol'], df['demrow'] = ~a * (df['e'], df['n'])
        
        # use map_coordinates to do bilinear interp and place result in new df column
        # need to transpose to get into rows to place into df
        df['dem_z'] = np.transpose(
            ndimage.map_coordinates(self.array, [[df['demrow']], [df['demcol']]], order=1, mode='constant', cval=-9999))
        
        # drop rows which are nan
        df.dropna(axis=0, subset=['dem_z'], inplace=True)
        
        # drop rows which were assigned constant -9999 (outside of dem bounds)
        df = df.loc[df['dem_z'] != -9999]
        
        # calculate residual (obs - pred), or (check-dem)
        df['resid'] = df['gps_z'] - df['dem_z']
        
        # Calc RMSE, mean, mae, stdev
        rmse = ((df['gps_z'] - df['dem_z']) ** 2).mean() ** .5
        mean_error = df['resid'].mean()
        mae = df['resid'].abs().mean()
        stdev = df['resid'].std()
        
        outdict = {'rmse':rmse, 'mean_error':mean_error, 'mae':mae, 'stdev':stdev}
        
        return(outdict, df)
        
        



