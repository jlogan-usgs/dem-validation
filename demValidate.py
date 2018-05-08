"""
Created on Tue May  1 14:48:07 2018

Initial script to do simple DEM validation

@author: jlogan
"""

import rasterio 
import numpy as np
import pandas as pd
from scipy import ndimage
import seaborn as sns
import matplotlib.pyplot as plt
import argparse

#Define input files here or in command line.  Command line arguments are used if both provided
#path to DEM (has to be geotiff)
demfileconst = 'D:\\UAS\\2018-605-FA\\products\\DEM\\DEM_GrndClass\\2018-04-ClvCorral_DEM_5cm.tif'

#path to check points csv (needs to have columns 'n, 'e', 'z')
checkfileconst = 'D:\\UAS\\2018-605-FA\\GPS\\2018-04-ClvCorral_RTK_Validation_nez.csv'

#plot
errorplotconst=True

#=======================PARSE ARGUMENTS======================
descriptionstr = ('  Script to validate DEMs using check points from csv file')
parser = argparse.ArgumentParser(description=descriptionstr, 
                                 epilog='example: demValidate.py -dem mygeotiff.tif -checkpointfile mycheckpointfile.csv -plot=True -map=True')
parser.add_argument('-dem', '--dem', dest='demfile', nargs='?', const='undefined', type=str, 
                    help='DEM geotiff')
parser.add_argument('-checkpoints','--checkpointfile', dest='checkfile', nargs='?', const='undefined', type=str, 
                    help='Comma delimited text file with check points, needs header with n, e, z')
parser.add_argument('-plot','--errorplot', dest='errorplot', nargs='?', const=True, type=bool, 
                    help='Plot error distribution plot [boolean]')
parser.add_argument('-map','--mapplot', dest='mapplot', nargs='?', const=True, type=bool, 
                    help='Show plot of hillshade with check points [boolean]')
args = parser.parse_args()

#check arguments
if args.demfile is not None:
    #Then use command line argument
    demfile = args.demfile
    #remove quotes in string if supplied
    demfile = demfile.replace('"', '').replace("'", '')
else:
    #use demfilefileconst from top of script
    demfile = demfileconst
print('Input DEM: ' + demfile)

if args.checkfile is not None:
    #Then use command line argument
    checkfile = args.checkfile
    #remove quotes in string if supplied
    checkfile = checkfile.replace('"', '').replace("'", '')
else:
    #use demfilefileconst from top of script
    checkfile = checkfileconst
print('Input check point file: ' + checkfile)

if args.errorplot is not None:
    #Then use command line argument
    errorplot = args.errorplot
else:
    #use errorplotconst from top of script
    errorplot = errorplotconst


########################################
#Main script below

#load DEM (geotiff)
dataset = rasterio.open(demfile)

#get numpy array
dem = dataset.read(1)

#convert nodatavalues to nans
dem[dem==dataset.nodatavals] = np.nan

# #### How to use affine transform, from https://www.perrygeo.com/python-affine-transforms.html
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
mean_error = df['resid'].mean()
mae = df['resid'].abs().mean()
stdev = df['resid'].std()


#print results
print('RMSE: ' + str(rmse))
print('Mean offset: ' + str(mean_error))
print('Std. Dev.: ' + str(stdev))
print('MAE: ' + str(mae))

#Plot histogram?
if errorplot:
    #Then plot histogram
    #set seaborn style
    sns.set_style('darkgrid')
    f = plt.figure(figsize=(7.5,5))
    
    ax =sns.distplot(df['resid'], bins=50, kde=False, hist_kws=dict(edgecolor="b", linewidth=0.5))
    #set xlimit to be equal on either side of zero
    ax.set_xlim(np.abs(np.array(ax.get_xlim())).max()*-1, np.abs(np.array(ax.get_xlim())).max())
    #plot vertical line at 0
    ax.axvline(x=0, color='k', linestyle='--', alpha=0.8, linewidth=0.8)
    
    #make annotation str
    s = ('RMSE: ' + "{:0.3f}".format(rmse) + 'm' + '\n' +
          'Mean Error: ' + "{:0.3f}".format(mean_error) + 'm' + '\n' +
          'Std. Dev. of Error: ' + "{:0.3f}".format(stdev) + 'm' + '\n' +
          'Mean Error: ' + "{:0.3f}".format(mean_error) + 'm')
    #place text at 40% on right, 80% top
    ax.text(np.abs(np.array(ax.get_xlim())).max()*0.4, np.array(ax.get_ylim()).max()*0.8, s, alpha=0.8, fontsize=10)
    
    ax.set_xlabel('Elevation difference, GPS - DEM [m]')
    ax.set_ylabel('count [n]')
    
    f.suptitle('DEM Validation', fontstyle='italic')





