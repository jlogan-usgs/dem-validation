
# coding: utf-8

# # Initial tests to run DEM validation

# In[17]:


import rasterio 
import numpy as np
import pandas as pd
from scipy import ndimage


# In[4]:


#path to DEM
demfile = 'C:\\jlogan_python\\demValidation\\data\\2017-1101-LPD_UAS-SfM-DEM_10cm.tif'

#load DEM (geotiff)
dataset = rasterio.open(demfile)


# In[19]:


#get numpy array
dem = dataset.read(1)


# #### How to use affine transform, from https://www.perrygeo.com/python-affine-transforms.html
# 
# #### Using rasterio and affine
# `a = ds.affine`
# #### col, row to x, y
# `x, y = a * (col, row)`
# #### x, y to col, row
# `col, row = ~a * (x, y)`

# In[8]:


#get affine transform
a = dataset.affine


# In[13]:


#Test affine

#Top left coords of dataset should be 0,0
col, row = ~a * (dataset.bounds.left, dataset.bounds.top)
print(str(col) + ', ' + str(row))


# In[20]:


#Test getting array coords with geocoords from validation file
north = 4026533.048
east = 619859.129
col, row = ~a * (east, north)
#print(str(col) + ', ' + str(row))
~a * (east, north)
#need to check on order of x,y and col, row


# In[44]:


#try map_coordinates bilinear interp

#first get indices into array
coords = np.asarray([1, 1])
z = ndimage.map_coordinates(dem, [[row],[col]], order=1, mode='constant', cval=-9999)
print(str(z))


# In[37]:


dem[5103,3179]


# In[40]:


z.shape

