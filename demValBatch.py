"""
Created on Wed May  30 14:20:00 2018

Script to do DEM validation on a batch of DEM files using dem_validate function

@author: jlogan
"""

import sys
import os
import pandas as pd
import seaborn as sns
import numpy as np
import glob
import pathlib
import matplotlib.pyplot as plt
from matplotlib.colors import LightSource
from matplotlib.transforms import BboxBase as bbase
from tqdm import tqdm


# import DEM val functions
pathtodemvalscript = 'D:\\jloganPython\\dem-validation'
sys.path.append(pathtodemvalscript)
from demValidate import dem_validate

# set inputs/outputs
demfiles = 'D:\\jloganPython\\dem-validation\\data\\batch\\ob\\dems\\*.tif'
checkcsv = 'D:\\jloganPython\\dem-validation\\data\\batch\\ob\\val_pts\\2017-0307-OB_ne_elht_NAD83_CORS96.csv'
outcheckdir = 'D:\\jloganPython\\dem-validation\\data\\batch\\ob\\results'
outplotdir = 'D:\\jloganPython\\dem-validation\\data\\batch\\ob\\results\\plots'
outmasterresultsfile = 'DEMValMasterResults_no_dup.csv'
errorplot=True
mapplot=True


def custom_error_plot(valdf):
    """
    Function to plot distribution of residuals from dem validation.
    Using custom xlim settings instead of those from demValidate.py

    args:
        valdf: dataframe with gps_z, dem_z, and residual at each checkpoint.  Created by dem_validation function.

    returns:
        fig_hist: handle on plot object
    """

    # set seaborn style
    sns.set_style('darkgrid')
    fig_hist = plt.figure(figsize=(7.5, 5))

    ax = sns.distplot(valdf['resid'], bins=100, kde=False, hist_kws=dict(edgecolor="b", linewidth=0.5))
    # set xlimit to be equal on either side of zero
    # get max xlim first
    max_xlim = np.abs(np.array(ax.get_xlim())).max()
    if max_xlim >= 4:
        max_xlim = 4
    ax.set_xlim(max_xlim * -1, max_xlim)
    # plot vertical line at 0
    ax.axvline(x=0, color='k', linestyle='--', alpha=0.8, linewidth=0.8)

    # Calc RMSE, mean, mae, stdev from resid for plot annotation (these were already calculated in dem_validate function
    # but recalculating here to reduce dependency
    rmse = ((valdf['gps_z'] - valdf['dem_z']) ** 2).mean() ** .5
    mean_error = valdf['resid'].mean()
    mae = valdf['resid'].abs().mean()
    stdev = valdf['resid'].std()

    # make annotation str
    s = ('RMSE:                   ' + "{:0.3f}".format(rmse) + 'm' + '\n' +
         'Mean Error:          ' + "{:0.3f}".format(mean_error) + 'm' + '\n' +
         'Std. Dev. of Error: ' + "{:0.3f}".format(stdev) + 'm' + '\n' +
         'MAE:                     ' + "{:0.3f}".format(mae) + 'm' + '\n' +
         'n:                           ' + str(len(valdf)))
    # place text at 40% on right, 80% top
    ax.text(np.abs(np.array(ax.get_xlim())).max() * 0.4, np.array(ax.get_ylim()).max() * 0.8, s, alpha=0.8, fontsize=10)

    ax.set_xlabel('Elevation difference, GPS - DEM [m]')
    ax.set_ylabel('count [n]')

    fig_hist.suptitle('DEM Validation', fontstyle='italic')
    plt.show()

    return fig_hist


def custom_plot_map(dem, valdf):
    """
    Function to plot hillshade map of dem and checkpoints colored by residual.
    Using custom settings instead of plot_map from demValidate.py

    args:
        dem: numpy array of dem. Returned by dem_validation function.
        valdf: dataframe with gps_z, dem_z, and residual at each checkpoint.  Returned by dem_validation function.

    returns:
        fig_map: handle on plot object
    """
    # reset seaborn
    sns.reset_orig()
    ltsrc = LightSource(azdeg=315, altdeg=45)
    fig_map = plt.figure(figsize=(9, 9))
    plt.imshow(ltsrc.hillshade(dem, vert_exag=1.5, dx=0.1, dy=0.1), cmap='gray')

    # plot points, using img coords, colors as resid.mean +- 3 stddev
    #plt.scatter(x=valdf['demcol'], y=valdf['demrow'], c=valdf['resid'], cmap=plt.cm.jet, s=2, alpha=0.8)
    plt.scatter(x=valdf['demcol'], y=valdf['demrow'], c=valdf['resid'], cmap=plt.cm.jet, 
                s=2, alpha=0.5, 
                vmin=valdf['resid'].mean()-(3*valdf['resid'].std()), 
                vmax=valdf['resid'].mean()+(3*valdf['resid'].std()))
#    plt.xlim(valdf['demcol'].min(), valdf['demcol'].max())
#    plt.ylim(valdf['demrow'].min(), valdf['demrow'].max())
    
    plt.colorbar(aspect=45, pad=0.04, label='residual [m]')
    plt.show()

    return fig_map

def squeeze_fig_aspect(fig, preserve='h'):
    preserve = preserve.lower()
    bb = bbase.union([ax.bbox for ax in fig.axes])

    w, h = fig.get_size_inches()
    if preserve == 'h':
        new_size = (h * bb.width / bb.height, h)
    elif preserve == 'w':
        new_size = (w, w * bb.height / bb.width)
    else:
        raise ValueError(
            'preserve must be "h" or "w", not {}'.format(preserve))
    fig.set_size_inches(new_size, forward=True)
    

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
    
    # run dem_validate, one point per cell
    valstats, valdf, tmpdem, _ = dem_validate(file, checkcsv, outcheckfile, one_pt_per_cell=True)

    # load valstats to master dataframe using dict keys to map to columns
    for key, value in valstats.items():
        masterdf.loc[index, key] = value

    masterdf.loc[index, 'dem'] = demname
    masterdf.loc[index, 'n_pts'] = len(valdf)
    
    if errorplot:
        # do custom plot to keep x range between -4 and 4
        fig = custom_error_plot(valdf)
        # change title
        fig.suptitle(basedemname)
        fig.savefig((outplotdir + '\\' + basedemname + '_val_plot.png'), dpi=100)
        plt.close(fig)
        
    if mapplot:
        # do custom map with small pts
        mapfig = custom_plot_map(tmpdem, valdf)
        squeeze_fig_aspect(mapfig)
        plt.tight_layout()
        #mapfig.suptitle('GPS - DEM Residual')
        mapfig.savefig((outplotdir + '\\' + basedemname + '_val_map.png'), dpi=200)
        plt.close(mapfig)

# export masterdf to csv
# convert to col to float first
for col in ['rmse', 'mean', 'stdev', 'mae', 'max_abs_error', 'n_pts']:
    masterdf[col] = pd.to_numeric(masterdf[col])
    
masterdf.to_csv((outcheckdir + '\\' + outmasterresultsfile), float_format='%.4f', index=False)






