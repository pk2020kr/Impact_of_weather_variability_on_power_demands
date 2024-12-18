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


# Rainfall and Temperature
ind_geom = gpd.read_file('India_State_Boundary.shp')
up_geom = ind_geom[ind_geom['State_Name'] == 'Uttar Pradesh']

imd_r = imd.open_data('rain', 2020, 2020,'yearwise') # (variable, start_yr, end_yr, fn_format='yearwise')
imd_ra = imd_r.get_xarray()
imd_rai = imd_ra.where(imd_ra['rain'] != -999.) #Remove NaN values
imd_rain = imd_rai['rain'].max('time')
imd_rain.plot()


# Load rain data from IMD
imd_r = imd.open_data('rain', 2017, 2023, 'yearwise')  # (variable, start_yr, end_yr, fn_format='yearwise')
imd_ra = imd_r.get_xarray()
imd_rai = imd_ra.where(imd_ra['rain'] != -999.)  # Remove NaN values
imd_rain = imd_rai['rain'].mean('time')  # Calculate mean rainfall over the year
# Ensure rain data has geospatial coordinates
imd_rain = imd_rain.rio.write_crs(4326, inplace=True)  # Assume WGS84 lat/lon
# Reproject Uttar Pradesh boundary to match the rain data's CRS
up_geom = up_geom.to_crs(imd_rain.rio.crs)
ind_geom = ind_geom.to_crs(imd_rain.rio.crs)  # Ensure whole country is also in the same CRS
# Clip/mask the rain data to Uttar Pradesh geometry
rain_masked = imd_rain.rio.clip(up_geom.geometry, up_geom.crs, drop=False)  # Keep NaN outside UP
# Set up the plot
fig, ax = plt.subplots(figsize=(10, 10))
# Plot the entire country boundary
ind_geom.boundary.plot(ax=ax, color='black', linewidth=0.5, linestyle='--')
# Plot Uttar Pradesh boundary (as an overlay)
up_geom.boundary.plot(ax=ax, color='red', linewidth=1.5)
# Plot the clipped rain data for Uttar Pradesh
rain_plot = rain_masked.plot(ax=ax, cmap='Blues', alpha=0.7, add_colorbar=False)
# Add a colorbar for rain data
cbar = fig.colorbar(rain_plot, ax=ax, orientation='vertical')
cbar.set_label('Rainfall (mm)')
# Add a custom legend for the country boundary and Uttar Pradesh boundary
from matplotlib.lines import Line2D
legend_elements = [Line2D([0], [0], color='black', lw=0.5, linestyle='--', label='Country Boundary'),
                   Line2D([0], [0], color='red', lw=1.5, label='Uttar Pradesh Boundary')]
# Display the custom legend
ax.legend(handles=legend_elements, loc='upper right')
# Title and labels
ax.set_title('Rainfall in Uttar Pradesh (2017_2023) within India Map', fontsize=15)
ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')
plt.show()


# Ensure rain data has geospatial coordinates
imd_rain = imd_rain.rio.write_crs(4326, inplace=True)  # Assume WGS84 lat/lon
# Reproject Uttar Pradesh boundary to match the rain data's CRS
up_geom = up_geom.to_crs(imd_rain.rio.crs)
# Clip/mask the rain data to Uttar Pradesh geometry
rain_masked = imd_rain.rio.clip(up_geom.geometry, up_geom.crs, drop=True)
# Plot the result
fig, ax = plt.subplots(figsize=(10, 10))
# Plot the clipped rain data for Uttar Pradesh
rain_masked.plot(ax=ax, cmap='Blues', alpha=0.8)
# Highlight Uttar Pradesh boundary
up_geom.boundary.plot(ax=ax, color='red', linewidth=1.5)
# Title and labels
ax.set_title('Rainfall in Uttar Pradesh (2017_2023)', fontsize=16)
ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')
plt.show()


import geopandas as gpd
import pandas as pd
import rioxarray  # for working with raster data and masking
import xarray as xr
# Load India boundary shapefile and extract Uttar Pradesh boundary
ind_geom = gpd.read_file('India_State_Boundary.shp')
up_geom = ind_geom[ind_geom['State_Name'] == 'Uttar Pradesh']
# Load rain data from IMD
imd_r = imd.open_data('rain', 2017, 2023, 'yearwise')  # (variable, start_yr, end_yr, fn_format='daywise')
imd_ra = imd_r.get_xarray()
imd_rai = imd_ra.where(imd_ra['rain'] != -999.)  # Remove NaN values
# Ensure rain data has geospatial coordinates
imd_rai = imd_rai.rio.write_crs(4326, inplace=True)  # Assume WGS84 lat/lon
# Reproject Uttar Pradesh boundary to match the rain data's CRS
up_geom = up_geom.to_crs(imd_rai.rio.crs)
# Initialize a list to store the daily averages
daily_averages = []
# Loop over each day in 2020 and calculate the average rainfall in Uttar Pradesh
for day in imd_rai.time:
    # Extract data for the current day
    daily_rain = imd_rai.sel(time=day)['rain']    
    # Clip/mask the daily rain data to Uttar Pradesh geometry
    rain_masked = daily_rain.rio.clip(up_geom.geometry, up_geom.crs, drop=False)    
    # Calculate the average rainfall within Uttar Pradesh for this day
    daily_avg = rain_masked.mean().item()  # Mean of the non-NaN grid points    
    # Append the date and daily average to the list
    daily_averages.append({'date': pd.to_datetime(day.values), 'average_rain': daily_avg})
# Convert the list of daily averages to a DataFrame
daily_avg_df = pd.DataFrame(daily_averages)
# Save the DataFrame to CSV
daily_avg_df.to_csv('daily_avg_rainfall_up_2017_to_2023.csv', index=False)
print("Daily average rainfall in Uttar Pradesh (2017 to 2023) saved to daily_avg_rainfall_up_2017_to_2023.csv'")


imd_r = imd.open_data('tmax', 2017, 2023, 'yearwise')  # (variable, start_yr, end_yr, fn_format='daywise')
imd_ra = imd_r.get_xarray()
imd_rai = imd_ra.where(imd_ra['tmax'] < 90)  # Remove NaN values
imd_rain = imd_rai['tmax'].mean('time')
imd_rain.plot()


# Load tmax data from IMD
imd_r = imd.open_data('tmax', 2017, 2023, 'yearwise')  # (variable, start_yr, end_yr, fn_format='daywise')
imd_ra = imd_r.get_xarray()
imd_rai = imd_ra.where(imd_ra['tmax'] < 90)  # Remove NaN values
imd_rain = imd_rai['tmax'].mean('time')
# Ensure rain data has geospatial coordinates
imd_rain = imd_rain.rio.write_crs(4326, inplace=True)  # Assume WGS84 lat/lon
# Reproject Uttar Pradesh boundary to match the rain data's CRS
up_geom = up_geom.to_crs(imd_rain.rio.crs)
# Clip/mask the rain data to Uttar Pradesh geometry
rain_masked = imd_rain.rio.clip(up_geom.geometry, up_geom.crs, drop=True)
# Plot the result
fig, ax = plt.subplots(figsize=(10, 10))
# Plot the clipped rain data for Uttar Pradesh
rain_masked.plot(ax=ax, cmap='coolwarm', alpha=0.8)
# Highlight Uttar Pradesh boundary
up_geom.boundary.plot(ax=ax, color='red', linewidth=1.5)
# Title and labels
ax.set_title('tmax in Uttar Pradesh (2017_2023)', fontsize=16)
ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')
plt.show()


# Load tmax data from IMD
imd_r = imd.open_data('tmax', 2017, 2023, 'yearwise')  # (variable, start_yr, end_yr, fn_format='daywise')
imd_ra = imd_r.get_xarray()
imd_rai = imd_ra.where(imd_ra['tmax'] < 90)  # Remove NaN values
# Ensure rain data has geospatial coordinates
imd_rai = imd_rai.rio.write_crs(4326, inplace=True)  # Assume WGS84 lat/lon
# Reproject Uttar Pradesh boundary to match the rain data's CRS
up_geom = up_geom.to_crs(imd_rai.rio.crs)
# Initialize a list to store the daily averages
daily_averages = []
# Loop over each day in 2020 and calculate the average rainfall in Uttar Pradesh
for day in imd_rai.time:
    # Extract data for the current day
    daily_rain = imd_rai.sel(time=day)['tmax']    
    # Clip/mask the daily rain data to Uttar Pradesh geometry
    rain_masked = daily_rain.rio.clip(up_geom.geometry, up_geom.crs, drop=False)   
    # Calculate the average rainfall within Uttar Pradesh for this day
    daily_avg = rain_masked.mean().item()  # Mean of the non-NaN grid points    
    # Append the date and daily average to the list
    daily_averages.append({'date': pd.to_datetime(day.values), 'average_tmax': daily_avg})
# Convert the list of daily averages to a DataFrame
daily_avg_df = pd.DataFrame(daily_averages)
# Save the DataFrame to CSV
daily_avg_df.to_csv('daily_avg_tmax_up_2017_to_2023.csv', index=False)
print("Daily average tmax in Uttar Pradesh (2017 to 2023) saved to daily_avg_tmax_up_2017_to_2023.csv")


# Below code show Scatter Plot of Average Tmax over Time with Best-Fit Line of up from 1951 to 2023 you can see how it changing
tmax = pd.read_csv('daily_avg_tmax_up_1951_to_2023.csv', parse_dates=['date'], index_col='date')
avg_year = tmax.resample('ME').mean()
# Calculate the best-fit line
x = np.arange(len(avg_year))
y = avg_year['average_tmax'].values
coefficients = np.polyfit(x, y, 1)  # Fit a linear polynomial (degree 1)
slope = coefficients[0]
intercept = coefficients[1]
# Generate the line equation
line_equation = f'y = {slope:.8f}x + {intercept:.2f}'
# Plot the data and best-fit line
plt.figure(figsize=(10, 6))
plt.scatter(avg_year.index, avg_year['average_tmax'], label='Data Points')
# Plot the best-fit line
plt.plot(avg_year.index, slope * x + intercept, color='red', label=line_equation)
plt.xlabel('Date')
plt.ylabel('Average Tmax')
plt.title('Scatter Plot of Average Tmax over Time with Best-Fit Line')
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()
plt.show()
print(f"Equation of the best-fit line:\n{line_equation}")
