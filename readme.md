# DEM validation

Script to validate DEM using check points (usually collected with RTK GPS, or other survey methods).  The script evaluates vertical differences between check points and DEM.  Scipy `ndimage.map_coordinates` is used to do a bilinear interpolation of DEM z values for each checkpoint.  RMSE and other statistics are calculated, and an error distribution plot and map showing checkpoints on a hillshade of the DEM are optionally created.

---

#### Files in repository

- demValidate.py:  Main python script.
- demValidate_function_demo.ipynb: notebook showing use of DEM validation functions.
- demValidate_initial_tests.ipynb: notebook for initial testing.
- demClass.py: class for DEM and validation.  Not used by demValidation.py

#### Usage

1. Can be run directly from the python console with command line arguments:

    ```
	run demValidate.py -dem 'D:\myDEM.tif' -checkpoints  			'D:\myCheckpointfile_nez.csv' -outcsv 'D:\myOutputfile.csv' -mapplot=True
    ```
2. Alternatively can be run from python IDE, in which case the constants in the script are used instead of the command line arguments:

    ```
    #path to DEM (has to be geotiff)
    demfileconst = 'D:\\myDEM.tif'

    #path to check points csv (needs to have columns 'n, 'e', 'z')
    checkfileconst = 'D:\\myCheckpointfile_nez.csv'

    #path to output csv file
    outfileconst = 'D:\\myOutputfile.csv'

    #plot error distribution plot? [default = True]
    errorplotconst=True

    #plot map? [default = False]
    mapplotconst=False
    ```
3. Functions can be imported and used in other scripts:

    ```
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
    ```


#### Requirements

- DEM (currently only supports GeoTIFF format)
- check point file in csv format with fields:
  - 'n': northing
  - 'e': easting
  - 'z': elevation

#### Dependencies

Requires the following python packages:

- rasterio 
- numpy
- pandas
- scipy
- argparse
- seaborn
- matplotlib

#### Future improvements

- Write function to thin checkpoints to 1 point per DEM cell.  This will help avoid bad statistics if many repeated points fall in one cell. 
- Create option to import DEMs in .asc grid format.
- Improve map display
  - Fix tick labels so they show DEM coords instead of numpy array coords.
  - Write something to resample grid before plotting if it's very big (otherwise it can bog down with large/high-res. grids)
- Make csv import smarter with respect to column names (so 'n','e','z' column names are more flexible).
- Include a function to read LAS files and find closest point (or separate script)?

#### Current Problems

- Command line setting of `--errorplot=F` doesn't work right
- Plotting map of large/high res. DEMs is slow.  Need to resample.
