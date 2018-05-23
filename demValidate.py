"""
Created on Tue May  1 14:48:07 2018

Script to do simple vertical DEM validation using bilinear interplation of DEM z values at check point locations.

Script will use a DEM class to hold various DEM attributes and methods

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

#-----------------------INPUT ARGS------------------------------
# Define input files here or in command line.  These values will be used if called as script (in main())
# Command line arguments are used preferentially if provided.
demfileconst = 'D:\\UAS\\2018-605-FA\\products\\DEM\\DEM_GrndClass\\2018-04-ClvCorral_DEM_5cm.tif'  # Path to DEM (has to be geotiff)

checkfileconst = 'D:\\UAS\\2018-605-FA\\GPS\\2018-04-ClvCorral_RTK_Validation_nez.csv'  # Path to check points csv (needs to have columns 'n, 'e', 'z')

outfileconst = 'D:\\UAS\\2018-605-FA\\GPS\\2018-04-ClvCorral_RTK_Validation_nez_DEMz.csv'  # Path to output csv file

errorplotconst = True  # Plot error distribution plot? [default = True]

mapplotconst = False  # Plot map? [default = False]
#---------------------------------------------------------------

#Set up DEM class
class dem(object):
    def __init__(self, demfile):
        with rasterio.open(demfile) as src:
            #load dataset, get attributes
            self.dataset = src
            self.bounds = self.dataset.bounds
            self.nodatavals = self.dataset.nodatavals
            self.crs = self.dataset.crs
            #load array (must be single band, additional bands ignored)
            self.array = self.dataset.read(1)
            self.array[self.array == self.dataset.nodatavals] = np.nan
                
        
    def validate(self, checkfile):
        ''' 
        Perform vertical validation on DEM using scipy bilinear interpolation
        at checkpoints in csv checkfile.  Checkfile must have columns, 'n','e','z'. 
        '''
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
        self.validation_stats = outdict
        self.validation_points_df = df
        
        return outdict, df

def parse_command_line_args():
    # =======================PARSE ARGUMENTS======================
    descriptionstr = ('  Script to validate DEMs using check points from csv file')
    parser = argparse.ArgumentParser(description=descriptionstr,
                                     epilog='example: demValidate.py -dem mygeotiff.tif -checkpointfile mycheckpointfile.csv -plot -map=False')
    parser.add_argument('-dem', '--dem', dest='demfile', nargs='?', const='undefined', type=str,
                        help='DEM geotiff')
    parser.add_argument('-checkpoints', '--checkpointfile', dest='checkfile', nargs='?', const='undefined', type=str,
                        help='Comma delimited text file with check points, needs header with n, e, z')
    parser.add_argument('-outcsv', '--outcsvfile', dest='outfile', nargs='?', const='undefined', type=str,
                        help='Output comma delimited text file with interpolated DEM values')
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
    pargdict = {'demfile': demfile,
                'checkfile': checkfile,
                'outfile': outfile,
                'errorplot': errorplot,
                'mapplot': mapplot
                }
    return pargdict


def main(pargdict):
    #load dem into class
    dem = dem(pargdict['demfile'])
    
    #run validation
    dem.validate(pargdict['checkfile'])
    
    #get valstats, valdf
    valstats = dem.validation_stats
    valdf = dem.validation_points_df
        
    # print results
    print('RMSE: ' + '{:0.3f}'.format(valstats['rmse']))
    print('Mean offset: ' + '{:0.3f}'.format(valstats['mean_error']))
    print('Std. Dev.: ' + '{:0.3f}'.format(valstats['stdev']))
    print('MAE: ' + '{:0.3f}'.format(valstats['mae']))
    
    # export
    valdf.drop(['demrow', 'demcol'], axis=1).to_csv(pargdict['outfile'], index=False, float_format='%0.3f')
    
    # Plot histogram?
    if pargdict['errorplot']:
        # Then plot histogram
        print('Plotting error distribution plot')
        # set seaborn style
        sns.set_style('darkgrid')
        fig_hist = plt.figure(figsize=(7.5, 5))
    
        ax = sns.distplot(valdf['resid'], bins=50, kde=False, hist_kws=dict(edgecolor="b", linewidth=0.5))
        # set xlimit to be equal on either side of zero
        ax.set_xlim(np.abs(np.array(ax.get_xlim())).max() * -1, np.abs(np.array(ax.get_xlim())).max())
        # plot vertical line at 0
        ax.axvline(x=0, color='k', linestyle='--', alpha=0.8, linewidth=0.8)
    
        # make annotation str
        s = ('RMSE:                   ' + "{:0.3f}".format(valstats['rmse']) + 'm' + '\n' +
             'Mean Error:          ' + "{:0.3f}".format(valstats['mean_error']) + 'm' + '\n' +
             'Std. Dev. of Error: ' + "{:0.3f}".format(valstats['stdev']) + 'm' + '\n' +
             'MAE:                     ' + "{:0.3f}".format(valstats['mae']) + 'm' + '\n' +
             'n:                           ' + str(len(valdf)))
        # place text at 40% on right, 80% top
        ax.text(np.abs(np.array(ax.get_xlim())).max() * 0.4, np.array(ax.get_ylim()).max() * 0.8, s, alpha=0.8, fontsize=10)
    
        ax.set_xlabel('Elevation difference, GPS - DEM [m]')
        ax.set_ylabel('count [n]')
    
        fig_hist.suptitle('DEM Validation', fontstyle='italic')
        plt.show()
    
    # Plot map?
    if pargdict['mapplot']:
        # Then plot map (hillshade)
        print('Plotting map')
        # reset seaborn
        sns.reset_orig()
        ls = LightSource(azdeg=315, altdeg=45)
        fig_map = plt.figure(figsize=(9, 9))
        plt.imshow(ls.hillshade(dem.array, vert_exag=1.5, dx=0.1, dy=0.1), cmap='gray')
    
        # plot points, using img coords, colors as abs(resid)
        plt.scatter(x=valdf['demcol'], y=valdf['demrow'], c=valdf['resid'].abs(), cmap=plt.cm.jet, s=12, alpha=0.5)
        plt.show()
        
if __name__ == "__main__":
    pargdict = parse_command_line_args()
    main(pargdict)
    
