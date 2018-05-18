# DEM validation

Script to validate DEM using check points (usually collected with RTK GPS, or other survey methods).  The script evaluates vertical differences between check points and DEM.  Scipy `ndimage.map_coordinates` is used to do a bilinear interpolation of DEM z values for each checkpoint.  RMSE and other statistics are calculated, and an error distribution plot and map showing checkpoints on a hillshade of the DEM are optionally created.

#### Files in repository

- demValidate.py:  Main python script.
- demValidate.ipynb: Jupyter notebook for testing.

#### Usage

Can be run from the python console with command line arguments:

```
run demValidate.py -dem 'D:\myDEM.tif' -checkpoints  'D:\myCheckpointfile_nez.csv' -outcsv 'D:\myOutputfile.csv' -mapplot=True
```

Alternatively can be run from python IDE, in which case the constants in the script are used instead of the command line arguments:

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