import numpy as np
import pandas as pd
from datetime import timedelta
import pickle
import streamlit as st
import plotly.express as px
from st_aggrid import AgGrid
st.set_page_config(page_title="ENM Platforma",layout="wide")
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


st.write("#### **Mapa situación estación meteorológica cabo Udra y puntos modelo WRF Meteogalicia**") 

#load algorithm file gust

algo_g_d0=pickle.load(open("enm_udra/gust_udr_d0.al","rb"))


#load raw meteorological model and get model variables
meteo_model=get_meteogalicia_model(algo_g_d0["coor"])

#map
px.set_mapbox_access_token("pk.eyJ1IjoiZ3JhbmFudHVpbiIsImEiOiJja3B4dGU4OTkwMTFmMm9ycnNhMjJvaGJqIn0.VWzx_PkD9A5cSUVsn_ijCA")
dist_map=px.scatter_mapbox(algo_g_d0["coor"], hover_data=['distance'],lat='lat', lon='lon',color='distance',
                           color_continuous_scale=px.colors.cyclical.IceFire,)
st.plotly_chart(dist_map)

#Select meteorological model wind features
w_g0=(meteo_model[:24].wind_gust0*1.94384).round(1).to_numpy()
dir0=(meteo_model[:24].dir0).round(0).to_numpy()

#select x _var
model_x_var_g=meteo_model[:24][algo_g_d0["x_var"]]

#forecast machine learning  gust knots
gust_ml=(algo_g_d0["ml_model"].predict(model_x_var_g)*1.94384).round(1)

#load algorithm file dir
algo_dir_d0=pickle.load(open("enm_udra/dir_udr_d0.al","rb"))

#select x _var
model_x_var_d=meteo_model[:24][algo_dir_d0["x_var"]]

#forecast machine learning wind direction degrees
dir_ml=algo_dir_d0["ml_model"].predict(model_x_var_d)

#compare results
df_show=pd.DataFrame({"Hora UTC":meteo_model[:24].index,
                      "Machine Learning dirección grados":dir_ml,
                      "Modelo WRF en punto 0 dirección grados":dir0,
                      "Machine Learning racha máxima nudos":gust_ml,
                      "Modelo WRF en punto 0 racha máxima nudos":w_g0,
                      })
                     
st.title(""" Pronóstico viento estación cabo Udra Modelo WRF de Meteogalicia y Machine Learning""")
AgGrid(df_show)
with open("enm_udra/Informe_calidad.pdf", "rb") as pdf_file:
    PDFbyte = pdf_file.read()

st.download_button(label="Descargar informe de calidad",
                    data=PDFbyte,
                    file_name="informe_calidad.pdf",
                    mime='application/octet-stream')

#load algorithm file dir
algo_prec_d1=pickle.load(open("algo_list/prec_mar_d1.al","rb"))

#select x _var
model_x_var_p=meteo_model[24:48][algo_prec_d1["x_var"]]

#forecast machine learning wind precipitation
prec_ml=algo_prec_d1["ml_model"].predict_proba(model_x_var_p)
df_show_pre=pd.DataFrame(prec_ml,columns=["no p","precipitación"])
df_show_pre["Hora UTC"]=meteo_model.index[24:48]
df_show_pre.drop(columns=["no p"])
st.title(""" Probabilidad precipitación ENM mañana con Machine Learning""")
AgGrid(df_show_pre)

