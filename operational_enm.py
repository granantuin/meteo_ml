# -*- coding: utf-8 -*-
"""
Created on Sat Feb  5 10:54:11 2022

@author: usuario
"""

import numpy as np
import pandas as pd
from datetime import timedelta
import pickle
import streamlit as st


def get_meteogalicia_model(coorde):
    """
    get meteogalicia model from algo coordenates

    Returns
    -------
    dataframe with meteeorological variables forecasted.

    """
    
    #defining url to get model from Meteogalicia server
    today=pd.to_datetime("today")
    head1="http://mandeo.meteogalicia.es/thredds/ncss/modelos/WRF_HIST/d03"
    head2=today.strftime("/%Y/%m/wrf_arw_det_history_d03")
    head3=today.strftime("_%Y%m%d_0000.nc4?")
    head=head1+head2+head3
 
    var1="var=dir&var=mod&var=wind_gust&var=mslp&var=temp&var=rh&var=visibility&var=lhflx"
    var2="&var=lwflx&var=conv_prec&var=prec&var=swflx&var=shflx&var=cape&var=cin&var=cfh&var=T850"
    var3="&var=cfl&var=cfm&var=cft&var=HGT500&var=HGT850&var=T500&var=snow_prec&var=snowlevel"
    var=var1+var2+var3
 
    f_day=(today+timedelta(days=2)).strftime("%Y-%m-%d") 
    tail="&time_start="+today.strftime("%Y-%m-%d")+"T01%3A00%3A00Z&time_end="+f_day+"T23%3A00%3A00Z&accept=csv"
 

    dffinal=pd.DataFrame() 
    for coor in list(zip(coorde.lat.tolist(),coorde.lon.tolist(),np.arange(0,len(coorde.lat.tolist())).astype(str))):
        dffinal=pd.concat([dffinal,pd.read_csv(head+var+"&latitude="+str(coor[0])+"&longitude="+str(coor[1])+tail,).add_suffix(str(coor[2]))],axis=1)    
 
    
    #filter all columns with lat lon and date
    dffinal=dffinal.filter(regex='^(?!(lat|lon|date).*?)')
 
    #remove column string between brakets
    new_col=[c.split("[")[0]+c.split("]")[-1] for c in dffinal.columns]
    for col in zip(dffinal.columns,new_col):
        dffinal=dffinal.rename(columns = {col[0]:col[1]})
 
    dffinal=dffinal.set_index(pd.date_range(start=today.strftime("%Y-%m-%d"), end=(today+timedelta(days=3)).strftime("%Y-%m-%d"), freq="H")[1:-1])  
    # select x variables from algo
        
         
    return dffinal 

#load algorithm file gust
algo_g_d0=pickle.load(open("C:\\Users\\usuario\\Desktop\\colab\\algorithms\\gust_udr_d0.al","rb"))


#load raw meteorological model
meteo_model=get_meteogalicia_model(algo_g_d0["coor"])[:24]
w_g0=(meteo_model.wind_gust0*1.94384).round(1).to_numpy()

#select x _var
model_x_var_g=meteo_model[algo_g_d0["x_var"]]

#forecast variables knots
forecast=(algo_g_d0["ml_model"].predict(model_x_var_g)*1.94384).round(1)

#compare results
df_show_g=pd.DataFrame({"Machine Learning nudos":forecast,"Modelo meteorológico nudos":w_g0,
                      "Hora UTC":meteo_model.index})
                     
st.set_page_config(page_title="ENM cabo Udra Racha máxima",layout="wide")
st.dataframe(df_show_g)

#load algorithm file dir
algo_dir_d0=pickle.load(open("C:\\Users\\usuario\\Desktop\\colab\\algorithms\\dir_udr_d0.al","rb"))


#load raw meteorological model
meteo_model=get_meteogalicia_model(algo_dir_d0["coor"])[:24]
dir0=(meteo_model.dir0).to_numpy()

#select x _var
model_x_var_d=meteo_model[algo_dir_d0["x_var"]]

#forecast variables knots
forecast=algo_dir_d0["ml_model"].predict(model_x_var_d)

#compare results
df_show=pd.DataFrame({"Machine Learning grados":forecast,"Modelo meteorológico grados":dir0,
                      "Hora UTC":meteo_model.index})
                     

st.dataframe(df_show)