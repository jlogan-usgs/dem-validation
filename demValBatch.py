"""
Created on Wed May  30 14:20:00 2018

Script to do DEM validation on a batch of DEM files using dem_validate function

@author: jlogan
"""

import sys
import os
import glob
import pathlib
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd

# set inputs/outputs
demfiles = 'D:\\pathToDEMs\\*.tif'
checkcsv = 'D:\\pathToValidationCSV\\validationCheckPoints.csv'
outcheckdir = 'D:\\outputPathForPointFiles'
outplotdir = 'D:\\outputPathForPointPlots'
outmasterresultsfile = 'DEMValMasterResults.csv'

# import DEM val functions
pathtodemvalscript = 'D:\\jloganPython\\dem-validation'
sys.path.append(pathtodemvalscript)
from demValidate import dem_validate, plot_error_dist

#create output directories if needed
pathlib.Path(outcheckdir).mkdir(parents=False, exist_ok=True)
pathlib.Path(outplotdir).mkdir(parents=False, exist_ok=True)

# create empty dataframe for validation stats
col_names = ['dem', 'rmse', 'mean', 'stdev', 'mae', 'max_abs_error', 'n_pts']
masterdf  = pd.DataFrame(columns = col_names)

# get list of files
filelist = glob.glob(demfiles)

# loop through files
for index, file in enumerate(tqdm(filelist)):
#for index, file in enumerate(filelist):
    # get DEM name
    demname = os.path.basename(file)
    # get base DEM name
    basedemname = os.path.splitext(demname)[0]
    # output file name
    outcheckfile = (outcheckdir + '\\' + basedemname + '_validation_pts.csv')
    
    # message
    print('\n')
    print(str(index) + '/' + str(len(filelist)) + '. Validating ' + demname)
    
    # run dem_validate
    valstats, valdf, _ , _ = dem_validate(file, checkcsv, outcheckfile)

    # load valstats to master dataframe using dict keys to map to columns
    for key, value in valstats.items():
        masterdf.loc[index, key] = value

    masterdf.loc[index, 'dem'] = demname
    masterdf.loc[index, 'n_pts'] = len(valdf)

    # plot
    fig = plot_error_dist(valdf)
    # change title
    fig.suptitle(basedemname)
    fig.savefig((outplotdir + '\\' + basedemname + '_val_plot.png'), dpi=100)
    plt.close(fig)

# export masterdf to csv
# convert to col to float first
for col in ['rmse', 'mean', 'stdev', 'mae', 'max_abs_error', 'n_pts']:
    masterdf[col] = pd.to_numeric(masterdf[col])
    
masterdf.to_csv((outcheckdir + '\\' + outmasterresultsfile), float_format='%.4f', index=False)






