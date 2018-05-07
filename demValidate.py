"""
Created on Tue May  1 14:48:07 2018

Initial script to do simple DEM validation

@author: jlogan
"""

import rasterio 
import numpy as np
import pandas as pd
from scipy import ndimage

#path to DEM (has to be geotiff)
demfile = 'D:\\UAS\\2018-605-FA\\products\\DEM\\DEM_GrndClass\\2018-04-ClvCorral_DEM_5cm.tif'

#path to check points csv (needs to have columns 'n, 'e', 'z')
checkfile = 'D:\\UAS\\2018-605-FA\\GPS\\2018-04-ClvCorral_RTK_Validation_nez.csv'

#load DEM (geotiff)
dataset = rasterio.open(demfile)

#get numpy array
dem = dataset.read(1)

#convert nodatavalues to nans
dem[dem==dataset.nodatavals] = np.nan

# #### How to use affine transform, from https://www.perrygeo.com/python-affine-transforms.html
# 
# #### Using rasterio and affine
# `a = ds.affine`
# #### col, row to x, y
# `x, y = a * (col, row)`
# #### x, y to col, row
# `col, row = ~a * (x, y)`

#get affine transform
a = dataset.affine

#load check points into dataframe
df = pd.read_csv(checkfile)
#rename z column to distinguish from dem
df.rename(columns={'z':'gps_z'}, inplace=True)

#use affine to get DEM row, column into df
df['demcol'], df['demrow'] = ~a * (df['e'], df['n'])

#use map_coordinates to do bilinear interp and place result in new df column
#need to transpose to get into rows to place into df
df['dem_z'] = np.transpose(ndimage.map_coordinates(dem, [[df['demrow']],[df['demcol']]], order=1, mode='constant', cval=-9999))

#drop rows which are nan
df.dropna(axis=0, subset=['dem_z'], inplace=True)

#drop rows which were assigned constant -9999 (outside of dem bounds)
df = df.loc[df['dem_z'] != -9999]


#calculate residual (obs - pred), or (check-dem)
df['resid'] = df['gps_z'] - df['dem_z']

#RMSE
rmse = ((df['gps_z'] - df['dem_z']) ** 2).mean() ** .5

#print results
print('Mean offset: ' + str(df['resid'].mean()))
print('Std. Dev.: ' + str(df['resid'].std()))
print('MAE: ' + str(df['resid'].abs().mean()))
print('RMSE: ' + str(rmse))






