# Impact_of_weather_variability_on_power_demands
I will use power usage data for Indian states and analyse the impact of weather variability (temperature and rainfall) on electricity consumption in these states. In particular, I will examine the impact of very hot and rainy days on power usage.


Data Sources

Electricity Data

Electricity Consumption and area of states are taken by Kaggle https://www.kaggle.com/datasets/twinkle0705/state-wise-power-consumption-in-india

Electricity Consumption taking per-day data state-wise from 2017-04-01 to 2023-12-31 2466 rows × 31 columns Link

Electricity Consumption taking hourly data state-wise from 1 Jan 2017 to 30 April 2024 From NITI Aayog https://iced.niti.gov.in/energy/electricity/distribution/national-level-consumption/load-curve 64248 rows × 33 columns 7 Years and 4 months Link

Population State population data I’m taking from the Census of India 2011 https://censusindia.gov.in/census.website/data/data-visualizations/PopulationSearch_PCA_Indicators AP (84580777) combined population was given in the Census of India 2011 Telangana 35,003,674 https://en.wikipedia.org/wiki/Demographics_of_Telangana Andhra Pradesh 49,577,103 https://en.wikipedia.org/wiki/Andhra_Pradesh

National Centers for Environmental Information https://www.ncei.noaa.gov/ https://www.ncei.noaa.gov/access/metadata/landing-page/bin/iso?id=gov.noaa.ncdc%3AC00861 https://www.ncei.noaa.gov/maps/daily/ only per day(max_min_ave) data is available All stations haven’t data of all time some are newly started and some are closed

NASA Prediction Of Worldwide Energy Resources (POWER) | Data Access Viewer (DAV) https://power.larc.nasa.gov/data-access-viewer/ hourly data of a coordinate is available

India Meteorological Department gridded rainfall and temperature (minimum and maximum) data https://imdlib.readthedocs.io/en/latest/Usage.html#reading-imd-datasets capable of downloading gridded rainfall and temperature (minimum and maximum) data. Data available only per day rain / tmax / tmin 1951-01-01 to 2023-12-31

Shapefile for Indian states and UT https://github.com/AnujTiwari/India-State-and-Country-Shapefile-Updated-Jan-2020