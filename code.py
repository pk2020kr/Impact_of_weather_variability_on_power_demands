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


# Cleening multiple files and merge using Dask
pip install dask
pip install dask[complete]
import dask.dataframe as dd

df = dd.read_csv('d17.csv') # raw data is like that Delhi [d17, d18, d19, d20, d21, d22, d23, d24] 
df.head(4)

df.tail(6)


# Reading and slicing the data using Dask
m17 = dd.read_csv('m17.csv').head(8760)
m18 = dd.read_csv('m18.csv').head(8760)
m19 = dd.read_csv('m19.csv').head(8760)
m20 = dd.read_csv('m20.csv').head(8760)
m21 = dd.read_csv('m21.csv').head(8760)
m22 = dd.read_csv('m22.csv').head(8760)
m23 = dd.read_csv('m23.csv').head(8760)
m24 = dd.read_csv('m24.csv').head(2904)
# Concatenating all data into one Dask DataFrame
m = dd.concat([m17, m18, m19, m20, m21, m22, m23, m24])
# Step 1: Extract the year from the "State" column
year_df = m['State'].str.extract(r'(\d{4})')
# Convert the extracted year to int and then assign it as a column
m['Year'] = year_df[0].astype(int)
# Step 2: Convert "Date" column to datetime format (with year 1900 as a placeholder)
m['DateTime'] = dd.to_datetime(m['Date'] + ' ' + m['Year'].astype(str), format='%d-%b %I%p %Y')
# Step 3: Drop unnecessary columns
maharashtra_cleaned = m[['DateTime', 'Hourly Demand Met (in MW)']]
# Step 4: Rename columns for the final format
maharashtra_cleaned = maharashtra_cleaned.rename(columns={'DateTime': 'date', 'Hourly Demand Met (in MW)': 'Maharashtra'})
# Compute the final result if necessary (optional)
maharashtra_cleaned = maharashtra_cleaned.compute()
# Display the result (optional)
print(maharashtra_cleaned)


# Reading and slicing the data using Dask
m17 = dd.read_csv('d17.csv').head(8760)
m18 = dd.read_csv('d18.csv').head(8760)
m19 = dd.read_csv('d19.csv').head(8760)
m20 = dd.read_csv('d20.csv').head(8760)
m21 = dd.read_csv('d21.csv').head(8760)
m22 = dd.read_csv('d22.csv').head(8760)
m23 = dd.read_csv('d23.csv').head(8760)
m24 = dd.read_csv('d24.csv').head(2904)
# Concatenating all data into one Dask DataFrame
m = dd.concat([m17, m18, m19, m20, m21, m22, m23, m24])
# Step 1: Extract the year from the "State" column
year_df = m['State'].str.extract(r'(\d{4})')
# Convert the extracted year to int and then assign it as a column
m['Year'] = year_df[0].astype(int)
# Step 2: Convert "Date" column to datetime format (with year 1900 as a placeholder)
m['DateTime'] = dd.to_datetime(m['Date'] + ' ' + m['Year'].astype(str), format='%d-%b %I%p %Y')
# Step 3: Drop unnecessary columns
delhi_cleaned = m[['DateTime', 'Hourly Demand Met (in MW)']]
# Step 4: Rename columns for the final format
delhi_cleaned = delhi_cleaned.rename(columns={'DateTime': 'date', 'Hourly Demand Met (in MW)': 'Delhi'})
# Compute the final result if necessary (optional)
delhi_cleaned = delhi_cleaned.compute()
# Display the result (optional)
print(delhi_cleaned)

# Merge the two DataFrames on the 'date' column
merged_data = dd.merge(delhi_cleaned, maharashtra_cleaned, on='date')
merged_data.head(4)

# Set the 'date' column as the index (creating a new DataFrame)
merged_data = merged_data.set_index('date')
merged_data.head(2)
