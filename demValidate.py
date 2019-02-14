"""
Created on Tue May  1 14:48:07 2018

Script to do DEM validation using bilinear interplation of DEM z values at check point locations.
Script can be called from command line, or functions can be imported to other scripts.

example usage:
    from command line:
        >> run demValidate.py -dem 'D:\myDEM.tif' -checkpoints  'D:\myCheckpointfile_nez.csv'
                -outcsv 'D:\myOutputfile.csv' -mapplot=True

    import in another script:
        #update path to allow import
        import sys
        sys.path.append('C:\path_to_directory_with_script')
        from demValidate import *

        # run dem_validation
        valstats, valdf, dem, aff = dem_validate(demfile, checkfile, outfile)

        # error distribution plot
        fig_hist = plot_error_dist(valdf)

        # map plot
        fig_map = plot_map(dem, aff, valdf)

@author: jlogan
"""

import rasterio
import numpy as np
import pandas as pd
from scipy import ndimage
import argparse

import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LightSource

# ================ DEFINE INPUT FILES HERE OR IN COMMAND LINE ======================
# (Command line arguments are used preferentially)
######INPUTS######
# path to DEM (has to be geotiff)
#demfileconst = 'D:\\UAS\\2018-605-FA\\products\\DEM\\DEM_GrndClass\\2018-04-ClvCorral_DEM_5cm_clip.tif'
# path to check points csv (must have columns 'n, 'e', 'z')
#checkfileconst = 'D:\\UAS\\2018-605-FA\\GPS\\2018-04-ClvCorral_RTK_Validation_nez.csv'
# path to output csv file
#outfileconst = 'D:\\UAS\\2018-605-FA\\GPS\\2018-04-ClvCorral_RTK_Validation_nez_DEMz.csv'

#Molokai UAS
demfileconst = r"T:\UAS\2018-617-FA\UAS\products\RefractionCorrection\Polar_SubaerialGCPs\lastoolcleaned\2018-0624-Molokai_SubAirGCP_LAStoolgrnd_uf_thin20cm_RefrCorrected_50cmgrid_lowest.tif"
checkfileconst = r"T:\UAS\2018-617-FA\UAS\products\RefractionCorrection\shoals_check_pts\mkshls_uascheck.txt"
outfileconst = r"T:\UAS\2018-617-FA\UAS\products\RefractionCorrection\Polar_SubaerialGCPs\lastoolcleaned\2018-0624-Molokai_SubAirGCP_LAStoolgrnd_uf_thin20cm_RefrCorrected_50cmgrid_shlsspotcheck_lowest.csv"


# one point per cell? [default = True]
oneptpercellconst = False
# plot error distribution plot? [default = True]
errorplotconst = True
# plot map? [default = False]
mapplotconst = False


# ================ DEFINE INPUT FILES HERE OR IN COMMAND LINE ======================

def dem_validate(demfile, checkfile, outfile, **kwargs):
    """
    Function to validate dem using a csv file with check points.  Performs bilinear interpolation on dem at each
    checkpoint using  scipy.ndimage.map_coordinates.  Only vertical residuals are calculated.
    Dem file should be geotiff format. Checkpoint file must have columns named
    'n','e', and 'z' (y coordinate, x coordinate, and z coordinate).  Keyword arg
    one_pt_per_cell=True will remove check points where more than one point falls in a DEM cell.
    For each pixel only the first point in the file will be kept.

    args:
        demfile: path to geotiff dem (single band only)
        checkfile: csv file 'n','e', and 'z' columns (with header)
        outfile: path to output csv file (input checkfile, plus dem value at each point)
    
    kwargs:
        one_pt_per_cell: boolean

    returns:
        valstats: dictionary with rmse, mean_offset, std_dev, mean_abs_error
        valdf: dataframe with input checkpoints, and dem value at each point
        dem: numpy array of dem (for use in plot_map function)
        aff: affine transform (for use in plot_map function)
    """

    # load DEM (geotiff)
    dataset = rasterio.open(demfile)

    # get numpy array
    dem = dataset.read(1)

    # convert nodatavalues to nans
    dem[dem == dataset.nodatavals] = np.nan

    # #### How to use affine transform, from https://www.perrygeo.com/python-affine-transforms.html
    # #### Using rasterio and affine
    # `a = ds.affine`
    # #### col, row to x, y
    # `x, y = a * (col, row)`
    # #### x, y to col, row
    # `col, row = ~a * (x, y)`

    # get affine transform
    aff = dataset.affine

    # load check points into dataframe
    valdf = pd.read_csv(checkfile)
    # rename z column to distinguish from dem
    valdf.rename(columns={'z': 'gps_z'}, inplace=True)

    # use affine to get DEM row, column into df
    valdf['demcol'], valdf['demrow'] = ~aff * (valdf['e'], valdf['n'])
    
    # remove points in cells where number of points in cell > 1, keeping only first point
    # if 'one_pt_per_cell=True'
    if kwargs.get('one_pt_per_cell'):
        print('Dropping points where > 1 point per cell...')
        initlen = len(valdf)
        #get integer index of rows, col (use round instead of floor to group into center of numpy cell (center of pixel))
        valdf['demcol_int'] = np.round(valdf['demcol'])
        valdf['demrow_int'] = np.round(valdf['demrow'])
        valdf.drop_duplicates(['demrow_int','demcol_int'],keep='first', inplace=True)
        valdf.drop(['demrow_int', 'demcol_int'], axis=1, inplace=True)
        print('Dropped ' + str(initlen - len(valdf)) + ' points where > 1 point per cell.')
    
    # use map_coordinates to do bilinear interp and place result in new df column
    # need to transpose to get into rows to place into df
    valdf['dem_z'] = np.transpose(
        ndimage.map_coordinates(dem, [[valdf['demrow']], [valdf['demcol']]], order=1, mode='constant', cval=-9999))

    # drop rows which are nan
    valdf.dropna(axis=0, subset=['dem_z'], inplace=True)

    # drop rows which were assigned constant -9999 (outside of dem bounds)
    valdf = valdf.loc[valdf['dem_z'] != -9999]

    # calculate residual (obs - pred), or (check-dem)
    valdf['resid'] = valdf['gps_z'] - valdf['dem_z']

    # Calc RMSE, mean, mae, stdev
    rmse = ((valdf['gps_z'] - valdf['dem_z']) ** 2).mean() ** .5
    mean_error = valdf['resid'].mean()
    mae = valdf['resid'].abs().mean()
    stdev = valdf['resid'].std(ddof=0)   #ddof=0 for pop. st.dev. (n)
    max_abs_error = valdf['resid'].abs().max()
    n = len(valdf)

    # print results
#    print('RMSE: ' + '{:0.3f}'.format(rmse))
#    print('Mean offset: ' + '{:0.3f}'.format(mean_error))
#    print('Std. Dev.: ' + '{:0.3f}'.format(stdev))
#    print('MAE: ' + '{:0.3f}'.format(mae))
#    print('Max Abs. Err: ' + '{:0.3f}'.format(max_abs_error))
#    print('n: ' + str(n))

    # make a dict to store validation stats
    valstats = {'rmse': rmse, 'mean': mean_error, 'stdev': stdev, 'mae': mae, 'max_abs_error': max_abs_error, 'n': n}

    # export dataframe to csv
    valdf.drop(['demrow', 'demcol'], axis=1).to_csv(outfile, index=False, float_format='%0.3f')

    # return dem and affine for use in map_plot if needed, otherwise just use "_" for these var in function call
    return valstats, valdf, dem, aff


def plot_error_dist(valdf):
    """
    Function to plot distribution of residuals from dem validation.

    args:
        valdf: dataframe with gps_z, dem_z, and residual at each checkpoint.  Created by dem_validation function.

    returns:
        fig_hist: handle on plot object
    """

    # set seaborn style
    sns.set_style('darkgrid')
    fig_hist = plt.figure(figsize=(7.5, 5))

    ax = sns.distplot(valdf['resid'], bins=50, kde=False, hist_kws=dict(edgecolor="b", linewidth=0.5))
    # set xlimit to be equal on either side of zero
    ax.set_xlim(np.abs(np.array(ax.get_xlim())).max() * -1, np.abs(np.array(ax.get_xlim())).max())
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


def plot_map(dem, valdf, aff):
    """
    Function to plot hillshade map of dem and checkpoints colored by residual.

    args:
        dem: numpy array of dem. Returned by dem_validation function.
        valdf: dataframe with gps_z, dem_z, and residual at each checkpoint.  Returned by dem_validation function.
        aff: affine transformation. Returned by dem_validation function.

    returns:
        fig_map: handle on plot object
    """
    # reset seaborn
    sns.reset_orig()
    ltsrc = LightSource(azdeg=315, altdeg=45)
    fig_map = plt.figure(figsize=(9, 9))
    plt.imshow(ltsrc.hillshade(dem, vert_exag=1.5, dx=0.1, dy=0.1), cmap='gray')

    # plot points, using img coords, colors as abs(resid)
    plt.scatter(x=valdf['demcol'], y=valdf['demrow'], c=valdf['resid'].abs(), cmap=plt.cm.jet, s=12, alpha=0.5)
    
    ax = plt.gca()
    
    # set ticklabels to easting, northing instead of image coords
    el = []
    nl = []
    for col in ax.get_xticks():
        e, _ = aff * (col, 0)
        el.append(int(round(e)))
        
    for row in ax.get_yticks():
        _, n = aff * (0, row)
        nl.append(int(round(n))) 
    
    #set top y ticklabel to ''
    nl[0],nl[1] = '',''
    ax.set_xticklabels(el)
    ax.set_yticklabels(nl)
    
    #set tick label formats
    plt.yticks(fontsize=8, rotation=90, verticalalignment='center')
    plt.xticks(fontsize=8)
    
    cbar = plt.colorbar(aspect=45, pad=0.04, label='residual [m]')
    cbar.set_label('residual [m]',fontsize=8)
    cbar.ax.tick_params(labelsize=8)
    
    plt.show()

    return fig_map


def parse_cl_args():
    """ Parse arguments from command line"""
    # =======================PARSE ARGUMENTS======================
    descriptionstr = '  Script to validate DEMs using check points from csv file'
    parser = argparse.ArgumentParser(description=descriptionstr,
                                     epilog='example: demValidate.py -dem mygeotiff.tif '
                                            '-checkpointfile mycheckpointfile.csv -plot -map=False')
    parser.add_argument('-dem', '--dem', dest='demfile', nargs='?', const='undefined', type=str,
                        help='DEM geotiff')
    parser.add_argument('-checkpoints', '--checkpointfile', dest='checkfile', nargs='?', const='undefined', type=str,
                        help='Comma delimited text file with check points, needs header with n, e, z')
    parser.add_argument('-outcsv', '--outcsvfile', dest='outfile', nargs='?', const='undefined', type=str,
                        help='Output comma delimited text file with interpolated DEM values')
    parser.add_argument('-one_pt_per_cell', '--one_pt_per_cell', dest='onepointpercell', nargs='?', const=True, type=bool,
                        help='Plot error distribution plot [boolean]')
    parser.add_argument('-errorplot', '--errorplot', dest='errorplot', nargs='?', const=True, type=bool,
                        help='Plot error distribution plot [boolean]')
    parser.add_argument('-mapplot', '--mapplot', dest='mapplot', nargs='?', const=False, type=bool,
                        help='Show plot of hillshade with check points [boolean]')
    args = parser.parse_args()

    # check arguments
    if args.demfile is not None:
        # Then use command line argument
        demfile = args.demfile
        # remove quotes in string if supplied
        demfile = demfile.replace('"', '').replace("'", '')
    else:
        # use demfilefileconst from top of script
        demfile = demfileconst
    print('Input DEM: ' + demfile)

    if args.checkfile is not None:
        # Then use command line argument
        checkfile = args.checkfile
        # remove quotes in string if supplied
        checkfile = checkfile.replace('"', '').replace("'", '')
    else:
        # use demfilefileconst from top of script
        checkfile = checkfileconst
    print('Input check point file: ' + checkfile)

    if args.outfile is not None:
        # Then use command line argument
        outfile = args.outfile
        # remove quotes in string if supplied
        outfile = outfile.replace('"', '').replace("'", '')
    else:
        # use outfileconst from top of script
        outfile = outfileconst
    print('Output csv file: ' + outfile)
    
    if args.onepointpercell is not None:
        # Then use command line argument
        onepointpercell = args.errorplot
    else:
        # use errorplotconst from top of script
        onepointpercell = oneptpercellconst
    print('Point thinning = ' + str(onepointpercell))
    
    if args.errorplot is not None:
        # Then use command line argument
        errorplot = args.errorplot
    else:
        # use errorplotconst from top of script
        errorplot = errorplotconst
    print('Plot error distribution plot = ' + str(errorplot))

    if args.mapplot is not None:
        # Then use command line argument
        mapplot = args.mapplot
    else:
        # use errorplotconst from top of script
        mapplot = mapplotconst
    print('Plot map = ' + str(mapplot))

    return demfile, checkfile, outfile, onepointpercell, errorplot, mapplot


def main():
    """ Operations if demValidate called directly as script. """
    # if called directly as script then
    # get command line arguments
    demfile, checkfile, outfile, onepointpercell, errorplot, mapplot = parse_cl_args()

    # run dem validation
    valstats, df, dem, aff = dem_validate(demfile, checkfile, outfile, one_pt_per_cell=onepointpercell)

    # error distribution plot?
    if errorplot:
        fig_hist = plot_error_dist(df)

    # map plot?
    if mapplot:
        fig_map = plot_map(dem, df, aff)


if __name__ == '__main__':
    # if called directly as script, execute main.
    main()
