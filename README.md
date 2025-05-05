# Impact of Weather Variability on Power Demands

This project analyzes the relationship between weather conditions (temperature and rainfall) and electricity consumption in Indian states, with a particular focus on Uttar Pradesh. The analysis explores how extreme weather events like very hot days and heavy rainfall affect power usage patterns.

## Project Overview

Climate change is increasingly affecting weather patterns across the globe, leading to more frequent extreme weather events. This project aims to quantify the relationship between weather variability and electricity consumption, providing insights that could be valuable for power grid management, energy policy planning, and climate adaptation strategies.

## Data Sources

### Weather Data
- **India Meteorological Department (IMD)** gridded rainfall and temperature data (1951-2023)
  - Daily maximum temperature (tmax)
  - Daily minimum temperature (tmin)
  - Daily rainfall

### Electricity Data
- **NITI Aayog** hourly state-wise electricity consumption data (2017-2023)
  - Source: https://iced.niti.gov.in/energy/electricity/distribution/national-level-consumption/load-curve

### Geographical Data
- State and country boundary shapefiles for India
  - Source: https://github.com/AnujTiwari/India-State-and-Country-Shapefile-Updated-Jan-2020

## Methodology

1. **Data Preprocessing**: 
   - Extraction of daily average temperature and rainfall data for Uttar Pradesh
   - Aggregation of hourly electricity consumption to daily averages
   - Detrending using 15-day rolling averages to remove seasonal patterns

2. **Analysis Approaches**:
   - Correlation analysis between weather variables and electricity demand
   - Linear regression to quantify relationships
   - Visualization of patterns and relationships

## Key Findings

The analysis reveals significant relationships between:
- Maximum temperature and electricity demand (positive correlation)
- Rainfall events and electricity consumption patterns
- Detrended weather and power demand show clear relationships, suggesting that weather variability directly impacts energy consumption beyond seasonal patterns

## Repository Structure

- `Impact of weather variability on power demand.ipynb`: Main Jupyter notebook containing the complete analysis
- `code.py`: Python script with key analysis functions
- `Data_source.md`: Detailed information about data sources
- `daily_avg_tmax_up_2017_to_2023.csv`: Daily average maximum temperature for Uttar Pradesh
- `daily_avg_tmin_up_2017_to_2023.csv`: Daily average minimum temperature for Uttar Pradesh
- `daily_avg_rainfall_up_2017_to_2023.csv`: Daily average rainfall for Uttar Pradesh
- Shapefiles: Geographical data for mapping and spatial analysis

## Requirements

The analysis requires the following Python libraries:
- pandas
- numpy
- matplotlib
- seaborn
- imdlib
- geopandas
- xarray
- rioxarray

## How to Use

1. Clone this repository
2. Install required dependencies using `pip install -r requirements.txt`
3. Open the Jupyter notebook to explore the analysis
4. The Python script (`code.py`) contains reusable functions for similar analyses

## Future Work

Potential extensions of this research include:
- Expanding the analysis to more states across India
- Incorporating more weather variables (e.g., humidity, wind speed)
- Forecasting future power demands based on climate projections
- Developing machine learning models to predict electricity consumption based on weather forecasts

## Contact

For questions or collaboration opportunities regarding this project, please open an issue in this repository.

## License

This project is available under the MIT License. 