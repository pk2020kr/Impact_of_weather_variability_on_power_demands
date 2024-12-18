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


# Temp
tmax = pd.read_csv('daily_avg_tmax_up_2017_to_2023.csv', parse_dates=['date'], index_col='date')
rolling_mean_15d_tmax = tmax['average_tmax'].rolling(window=15).mean()
tmax['UP_detrended'] = tmax['average_tmax'] - rolling_mean_15d_tmax
# Electricity
e_all = pd.read_csv('NITI_Aayog_all.csv', parse_dates=['date'], index_col='date')
avg_d = e_all.resample('d').mean()
avg_day = avg_d['2017-01-01':'2023-12-31']
ele = avg_day.resample('d').mean()
rolling_mean_15d_ele = avg_day['UP'].rolling(window=15).mean()
ele['UP_detrended'] = avg_day['UP'] - rolling_mean_15d_ele
# Prepare the data for the scatter plot
x = tmax['UP_detrended']
y = ele['UP_detrended']
# Ensure x and y have the same length by taking the common dates
# This will only include data where both tmax and ele have values
common_dates = x.index.intersection(y.index) 
x = x.loc[common_dates]
y = y.loc[common_dates]
# Remove NaN values
x = x.dropna()
y = y.dropna()
# Calculate the best-fit line
coefficients = np.polyfit(x, y, 1)
slope = coefficients[0]
intercept = coefficients[1]
line_equation = f"y = {slope:.2f}x + {intercept:.2f}"
# Generate points for the best fit line
x_line = np.linspace(min(x), max(x), 100)  # Generate 100 points for a smoother line
y_line = slope * x_line + intercept
# Create the scatter plot
plt.figure(figsize=(10, 6))
plt.scatter(x, y, label='Data Points')
plt.plot(x_line, y_line, color='red', label=line_equation)
# Add labels and title
plt.xlabel("tmax['UP_detrended']")
plt.ylabel("ele['UP_detrended']")
plt.title("Scatter Plot with Best-Fit Line")
plt.legend()
plt.grid(True)
# Show the plot
plt.show()
print(f"Equation of the best-fit line: {line_equation}")


# Temp
e_all = pd.read_csv('NITI_Aayog_all.csv', parse_dates=['date'], index_col='date')
avg_d = e_all.resample('d').mean()
avg_day = avg_d['2017-01-01':'2023-12-31']
ele = avg_day.resample('d').mean()
rolling_mean_15d_ele = avg_day['UP'].rolling(window=15).mean()
ele['UP_detrended'] = avg_day['UP'] - rolling_mean_15d_ele
# Electricity
rain = pd.read_csv('daily_avg_rainfall_up_2017_to_2023.csv', parse_dates=['date'], index_col='date')
rolling_mean_15d_rain = rain['average_rain'].rolling(window=15).mean()
rain['UP_detrended'] = rain['average_rain'] - rolling_mean_15d_rain
# Prepare the data for the scatter plot
x = rain['UP_detrended']
y = ele['UP_detrended']
# Ensure x and y have the same length by taking the common dates
common_dates = x.index.intersection(y.index)
x = x.loc[common_dates]
y = y.loc[common_dates]
# Remove NaN values
x = x.dropna()
y = y.dropna()
# Recalculate common dates after removing NaNs
common_dates = x.index.intersection(y.index)
x = x.loc[common_dates]
y = y.loc[common_dates]
# Calculate the best-fit line
coefficients = np.polyfit(x, y, 1)
slope = coefficients[0]
intercept = coefficients[1]
line_equation = f"y = {slope:.2f}x + {intercept:.2f}"
# Generate points for the best fit line
x_line = np.linspace(min(x), max(x), 100)
y_line = slope * x_line + intercept
# Create the scatter plot
plt.figure(figsize=(10, 6))
plt.scatter(x, y, label='Data Points')
plt.plot(x_line, y_line, color='red', label=line_equation)
# Add labels and title
plt.xlabel("rain['UP_detrended']")
plt.ylabel("ele['UP_detrended']")
plt.title("Scatter Plot with Best-Fit Line")
plt.legend()
plt.grid(True)
# Show the plot
plt.show()
print(f"Equation of the best-fit line: {line_equation}")


# To run the below function you need a file tmax to read and co_test to store the data
# Function to process temperature data of co-ordinate of India and plot the graph of mean_temp of assine time_gap
def plot_temperature(lat, lon, year_gap, file_dir='co_test'): # taken data from imdlib data (1951 to 2023) only that much is available
    data = imd.open_data('tmax'    ,1951,     2023,  'yearwise')
    data.to_csv('coordinate.csv', lat, lon, file_dir)  # This downloads the file
    # Construct the filename based on latitude and longitude
    file_name = f'coordinate_{lat:.2f}_{lon:.2f}.csv'    
    # Read the temperature data from the file
    l = pd.read_csv(f'{file_dir}/{file_name}')    
    # Convert the 'DateTime' column to a DatetimeIndex
    l['DateTime'] = pd.to_datetime(l['DateTime'])    
    # Use the lat/lon combination for the correct column name in the CSV
    col_name = f'{lat} {lon}'    
    # Create a DataFrame with DateTime as index and temperature as 'temp' column
    l = pd.DataFrame(list(l[col_name]), index=l['DateTime'], columns=['temp'])    
    # Resample data by the specified year gap
    resample_rule = f'{year_gap}YE'  # 'YE' means year-end resampling
    max_year = l.resample(resample_rule).mean()    
    # Plotting the data
    plt.figure(figsize=(24, 8))
    sns.barplot(x=max_year.index, y=max_year['temp'])
    plt.title(f'Max Temperatures Every {year_gap} Years for Location ({lat}, {lon})')
    plt.xlabel('Year')
    plt.ylabel('Max Temperature (°C)')
    plt.xticks(rotation=45)
    plt.show()
    print(max_year)
       

lat = 26.60
lon = 80.84
year_gap = 1
# Call the function to plot the temperature graph
plot_temperature(lat, lon, year_gap)
