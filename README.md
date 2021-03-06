## **Main menu** 

Streamlit python files to deploy algorithm files with extension .al (algo_list directory)

#### [Link to the Streamlit Platform 4 Km resolution](https://share.streamlit.io/granantuin/meteo_ml/main/operational_st.py)
#### [Link to the Streamlit Platform 1 Km Baixas resolution](https://share.streamlit.io/granantuin/meteo_ml/main/operational_st_baixas.py)

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

##### **get_metar.ipynb**

Notebook to get meteorological data from METAR reports. [Database IOWA UNIVERSITY](https://mesonet.agron.iastate.edu/request/download.phtml?network=ES__ASOS)

## algo_list directory

Files containing algorithms. Format meteorologicalvariable-season-station-forecast(d0,d1...).al

##### **algo_builder.ipynb**

Notebook to create .al files. 

*Input data*: meteorological station files and meteorological model files

*Output data*: algorithm files (.al)

##### **algo_builder_functions.ipynb**

Notebook with help functions required by algo_builder notebook


