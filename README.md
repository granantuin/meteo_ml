
## **meteorological_models directory**
##### get_wrf.py
Python file to get the historic wrf model from Meteogalicia server.

*intput data:*.

lat_station = 42.626  latitude.

lon_station = -8.784  longitude.

n_nearest =  4  nearest points to the station.

forecast_D = 2 forecast day.

initial_day_YYYYMMDD = "20211201".

final_day_YYYYMMDD = "20211215".

spatial_resolution = "4Km".

*output data (examples)*

[kml file with meteorological model and station coordenates: lat42.58lon-8.8046p4R4KmD1.csv](https://github.com/granantuin/meteo_ml/blob/main/meteorological_models/lat42.58lon-8.8046p4R4KmD1.csv)

[csv file with met variables in columns:lat42.58lon-8.8046p4R4KmD1.csv](https://github.com/granantuin/meteo_ml/blob/main/meteorological_models/lat42.58lon-8.8046p4R4KmD1.csv)

[csv file dist_ with distances from meteorological points to the station: distan_lat42.58lon-8.8046p4R4Km.csv](https://github.com/granantuin/meteo_ml/blob/main/meteorological_models/distan_lat42.58lon-8.8046p4R4Km.csv)

##### get_wrf_1km.py

Python file to get the historic wrf model from Meteogalicia server 1km resolution

## **met_stations directory**

meteorological stations. Extension .zip and .pt (parquet file)

##### **get_metar**

Notebook 

files to deploy meteo machine learning to the web
## [Link to de Platform](https://share.streamlit.io/granantuin/meteo_ml/main/operational_st.py)
