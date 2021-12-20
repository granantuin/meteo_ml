
#@title Select station coordinates (N+ S- W- and E+) and n nearest points
lat_station = 42.58 #@param {type:"number"}
lon_station = -8.8046 #@param {type:"number"}
n_nearest =  4#@param {type:"integer"}
forecast_D = 3 #@param ["0", "1", "2", "3"] {type:"raw"}
initial_day_YYYYMMDD = "20211214" #@param {type:"raw"}
final_day_YYYYMMDD = "20211215" #@param {type:"raw"}
spatial_resolution = "4Km" #@param ["4Km", "12Km", "36Km"] {allow-input: true}


import simplekml
from urllib.request import urlretrieve
import xarray as xr
import numpy as np
import pandas as pd
from math import radians, cos, sin, asin, sqrt
from datetime import timedelta

def haversine(lon1, lat1, lon2, lat2):
       lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
       dlon, dlat = lon2 - lon1 ,lat2 - lat1 
       a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
       km = 2 * asin(sqrt(a))*6367
       return km
#directory drive
root=""

# select url from 4km, 12km, 1.3Km and 36km in order to get the nearest points. Not real forecast !!!
#url1="http://mandeo.meteogalicia.es/thredds/ncss/modelos/WRF_HIST/d05/2020/11/wrf_arw_det_history_d05_20211101_0000.nc4?var=mod&disableLLSubset=on&disableProjSubset=on&horizStride=1&time_start=2020-11-01T01%3A00%3A00Z&time_end=2020-11-01T01%3A00%3A00Z&timeStride=1&addLatLon=true&accept=netcdf"
url4="http://mandeo.meteogalicia.es/thredds/ncss/modelos/WRF_HIST/d03/2020/11/wrf_arw_det_history_d03_20201101_0000.nc4?var=mod&disableLLSubset=on&disableProjSubset=on&horizStride=1&time_start=2020-11-01T01%3A00%3A00Z&time_end=2020-11-01T01%3A00%3A00Z&timeStride=1&addLatLon=true&accept=netcdf"
url12="http://mandeo.meteogalicia.es/thredds/ncss/modelos/WRF_HIST/d02/2016/09/wrf_arw_det_history_d02_20160927_0000.nc4?var=mod&disableLLSubset=on&disableProjSubset=on&horizStride=1&time_start=2016-09-27T01%3A00%3A00Z&time_end=2016-09-27T01%3A00%3A00Z&timeStride=1&addLatLon=true&accept=netcdf"
url36="http://mandeo.meteogalicia.es/thredds/ncss/modelos/WRF_HIST/d01/2018/10/wrf_arw_det_history_d01_20181031_0000.nc4?var=mod&disableLLSubset=on&disableProjSubset=on&horizStride=1&time_start=2018-10-31T01%3A00%3A00Z&time_end=2018-10-31T01%3A00%3A00Z&timeStride=1&addLatLon=true&accept=netcdf"
resolution={"4Km":url4,"12Km":url12,"36Km":url36}
resol_tag={"4Km":"d03","12Km":"d02","36Km":"d01"}
df=xr.open_dataset(urlretrieve(resolution[spatial_resolution])[0],engine="netcdf4").to_dataframe()
df_n=pd.DataFrame(df[["lat","lon","mod"]].values,columns=df[["lat","lon","mod"]].columns)

#add station coordinates
df_n["lat_st"],df_n["lon_st"]=lat_station,lon_station

#find the distances from the meteorological model to the station 
for index, row in df_n.iterrows():
       df_n.loc[index, 'distance'] = round(haversine(row['lon'], row['lat'], row['lon_st'], row['lat_st']),2)

#select n nearest points to the station. 
df_r=df_n.sort_values(by=["distance"]).head(n_nearest)

# KML with all the distances to the station
#df_n["distance"]=df_n["distance"].astype(str)
#kml = simplekml.Kml()
#df_n.apply(lambda X: kml.newpoint(name=X["distance"], coords=[( X["lon"],X["lat"])]) ,axis=1)
#kml.save("full_"+spatial_resolution+".kml")  

# KML with nearest n points
df_r["distance"]=df_r["distance"].astype(str)
kmlr = simplekml.Kml()
df_r.apply(lambda X: kmlr.newpoint(name=X["distance"], coords=[( X["lon"],X["lat"])]) ,axis=1)
kmlr.newpoint(name="STATION",coords=[(lon_station,lat_station)])
kmlr.save(root+"lat"+str(lat_station)+"lon"+str(lon_station)+"p"+str(n_nearest)+"R"+str(spatial_resolution)+".kml")

#save nearest points as csv file
df_r.set_index(np.arange(0,n_nearest)).drop(["mod"],axis=1).to_csv(root+"distan_"+"lat"+str(lat_station)+"lon"+str(lon_station)+"p"+str(n_nearest)+"R"+str(spatial_resolution)+".csv")

#get the meteorological model
df_sum=pd.DataFrame()
for a_day in pd.date_range(start=str(initial_day_YYYYMMDD), end=str(final_day_YYYYMMDD)):
  print("analysis date:",a_day)
  head=a_day.strftime("http://mandeo.meteogalicia.es/thredds/ncss/modelos/WRF_HIST/"+resol_tag[spatial_resolution]+"/%Y/%m/wrf_arw_det_history_"+resol_tag[spatial_resolution]+"_%Y%m%d_0000.nc4?")
  f_day=(a_day+timedelta(days=forecast_D)).strftime("%Y-%m-%d") 
  var="var=dir&var=snow_prec&var=snowlevel&var=mod&var=wind_gust&var=mslp&var=temp&var=rh&var=visibility&var=lhflx&var=lwflx&var=conv_prec&var=prec&var=swflx&var=shflx&var=cape&var=cin&var=cfh&var=cfl&var=cfm&var=cft&var=HGT500&var=HGT850&var=T500&var=T850"
  tail="&time_start="+f_day+"T00%3A00%3A00Z&time_end="+f_day+"T23%3A00%3A00Z&accept=csv"
  try:
     dffinal=pd.DataFrame() 
     for coor in list(zip(df_r["lat"].astype(str),df_r["lon"].astype(str),np.arange(0,n_nearest).astype(str))):
        dffinal=pd.concat([dffinal,pd.read_csv(head+var+"&latitude="+coor[0]+"&longitude="+coor[1]+tail,).add_suffix(coor[2])],axis=1)

     #filter all columns with lat lon and date
     dffinal=dffinal.filter(regex='^(?!(lat|lon|date).*?)')
 
     #remove column string between brakets
     new_col=[c.split("[")[0]+c.split("]")[-1] for c in dffinal.columns]
     for col in zip(dffinal.columns,new_col):
       dffinal=dffinal.rename(columns = {col[0]:col[1]})
 
     #add time  
     if forecast_D==0:
       dffinal["time"]=pd.date_range(f_day,periods=24, freq='H')[1:]
     else:
       dffinal["time"]=pd.date_range(f_day,periods=24, freq='H') 
     df_sum=pd.concat([df_sum,dffinal.set_index("time")])

  except:
     print(a_day,"failed")

#time index to column
df_sum.reset_index(inplace=True)

#save csv file
df_sum.to_csv(root+"lat"+str(lat_station)+"lon"+str(lon_station)+"p"+str(n_nearest)+"R"+str(spatial_resolution)+"D"+str(forecast_D)+".csv")



