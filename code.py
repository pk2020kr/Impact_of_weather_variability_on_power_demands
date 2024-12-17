import imdlib as imd
import geopandas as gpd
import numpy as np             
import pandas as pd             
import matplotlib.pyplot as plt 
import seaborn as sns
import rioxarray  # for working with raster data and masking
import xarray as xr


# Data Collection
# Rainfall and Temperature from IMD

# Download from IMD in .GRD format
#data = imd.get_data('tmax', 2017, 2023,'yearwise') #(variable, start_yr, end_yr, fn_format, file_dir)
#data = imd.get_data('tmin', 2017, 2023,'yearwise') #(variable, start_yr, end_yr, fn_format, file_dir)
#data = imd.get_data('rain',2017, 2023,'yearwise') #(variable, start_yr, end_yr, fn_format, file_dir)

# Read IMD_.GRD data    (variable,  start_yr, end_yr, fn_format, file_dir)
#data_tmax = imd.open_data('tmax'    ,2017,     2023,  'yearwise') # kept in same dir
#data_tmin = imd.open_data('tmin'    ,2017,     2023,  'yearwise') # kept in same dir
#data_rain = imd.open_data('rain'    ,2017,     2023,  'yearwise') # kept in same dir
