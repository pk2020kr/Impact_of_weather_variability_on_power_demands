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


df = pd.read_csv('d17.csv') # raw data is like that Delhi [d17, d18, d19, d20, d21, d22, d23, d24] 
df                                # similarly Maharashtra [m17, m18, m19, m20, m21, m22, m23, m24]

d17_ = df[0:8761]
d17_.tail(4)

d17 = df[0:8760]
d17.tail(2)

df = pd.read_csv('d24.csv')
d24 = df[0:2905]
d24.tail(4)

df = pd.read_csv('d17.csv')
d17 = df[0:8760]
df = pd.read_csv('d18.csv')
d18 = df[0:8760]
df = pd.read_csv('d19.csv')
d19 = df[0:8760]
df = pd.read_csv('d20.csv')
d20 = df[0:8760]
df = pd.read_csv('d21.csv')
d21 = df[0:8760]
df = pd.read_csv('d22.csv')
d22 = df[0:8760]
df = pd.read_csv('d23.csv')
d23 = df[0:8760]
df = pd.read_csv('d24.csv')
d24 = df[0:2904]
# Concatenating all data into one DataFrame
delhi = pd.concat([d17, d18, d19, d20, d21, d22, d23, d24], ignore_index=True)
delhi

# download the DataFrame(delhi) as a (delhi_ok.csv)as a file_name
#delhi.to_csv('delhi_ok.csv', index=False)

# Step 1: Extract the year from the "State" column
delhi['Year'] = delhi['State'].str.extract(r'(\d{4})').astype(int)
# Step 2: Convert "Date" column to datetime format (with year 1900 as a placeholder)
delhi['DateTime'] = pd.to_datetime(delhi['Date'] + ' ' + delhi['Year'].astype(str), format='%d-%b %I%p %Y')
# Step 3: Drop unnecessary columns
delhi_cleaned = delhi[['DateTime', 'Hourly Demand Met (in MW)']]
# Step 4: Rename columns for the final format
delhi_cleaned.columns = ['date', 'Delhi']
# Display the final DataFrame
delhi_cleaned


# Reading and slicing the data
m17 = pd.read_csv('m17.csv').iloc[:8760]
m18 = pd.read_csv('m18.csv').iloc[:8760]
m19 = pd.read_csv('m19.csv').iloc[:8760]
m20 = pd.read_csv('m20.csv').iloc[:8760]
m21 = pd.read_csv('m21.csv').iloc[:8760]
m22 = pd.read_csv('m22.csv').iloc[:8760]
m23 = pd.read_csv('m23.csv').iloc[:8760]
m24 = pd.read_csv('m24.csv').iloc[:2904]
# Concatenating all data into one DataFrame
m = pd.concat([m17, m18, m19, m20, m21, m22, m23, m24], ignore_index=True)
# Step 1: Extract the year from the "State" column
m['Year'] = m['State'].str.extract(r'(\d{4})').astype(int)
# Step 2: Convert "Date" column to datetime format (with year 1900 as a placeholder)
m['DateTime'] = pd.to_datetime(m['Date'] + ' ' + m['Year'].astype(str), format='%d-%b %I%p %Y')
# Step 3: Drop unnecessary columns
maharashtra_cleaned = m[['DateTime', 'Hourly Demand Met (in MW)']]
# Step 4: Rename columns for the final format
maharashtra_cleaned.columns = ['date', 'Maharashtra']
maharashtra_cleaned


# Reading and slicing the data
up17 = pd.read_csv('up17.csv').iloc[:8760]
up18 = pd.read_csv('up18.csv').iloc[:8760]
up19 = pd.read_csv('up19.csv').iloc[:8760]
up20 = pd.read_csv('up20.csv').iloc[:8760]
up21 = pd.read_csv('up21.csv').iloc[:8760]
up22 = pd.read_csv('up22.csv').iloc[:8760]
up23 = pd.read_csv('up23.csv').iloc[:8760]
up24 = pd.read_csv('up24.csv').iloc[:2904]
# Concatenating all data into one DataFrame
m = pd.concat([up17, up18, up19, up20, up21, up22, up23, up24], ignore_index=True)
# Step 1: Extract the year from the "State" column
m['Year'] = m['State'].str.extract(r'(\d{4})').astype(int)
# Step 2: Convert "Date" column to datetime format (with year 1900 as a placeholder)
m['DateTime'] = pd.to_datetime(m['Date'] + ' ' + m['Year'].astype(str), format='%d-%b %I%p %Y')
# Step 3: Drop unnecessary columns
up_cleaned = m[['DateTime', 'Hourly Demand Met (in MW)']]
# Step 4: Rename columns for the final format
up_cleaned.columns = ['date', 'up']
up_cleaned


# Merge the two DataFrames on the 'date' column
merged_data = pd.merge(delhi_cleaned, maharashtra_cleaned, on='date')
merged_data
# Merge the two DataFrames on the 'date' column
merged_data = pd.merge(merged_data, up_cleaned, on='date')
merged_data

from functools import reduce
# List of DataFrames to merge
dfs = [delhi_cleaned, maharashtra_cleaned, up_cleaned]
# Merge all DataFrames on 'date'
merged_data_all = reduce(lambda left, right: pd.merge(left, right, on='date'), dfs,)
# Set 'date' as the index
merged_data_all.set_index('date', inplace=True)
merged_data_all
