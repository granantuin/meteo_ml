# -*- coding: utf-8 -*-
"""
Created on Sun Sep  5 17:34:42 2021

@author: usuario
"""
import numpy as np
import pandas as pd
from datetime import timedelta
import pickle
import streamlit as st
import plotly.express as px
import os
import base64



def get_table_download_link(df):
    """Generates a link allowing the data in a given panda dataframe to be downloaded
    in:  dataframe
    out: href string
    """
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # some strings <-> bytes conversions necessary here
    return f'<a href="data:file/csv;base64,{b64}">Download csv file</a>' 
#load algorithm and select quality features



#algorithm list selection
st.set_page_config(page_title="Meterological Machine Learning Platform",layout="wide")
st.write("### **Algorithm  Format** ")
st.write("*meteorologicalvariablecode-location-forecastday.al*")

df_expla=pd.DataFrame({"meteorologicalvariablecode":["dir","spdb","fr"],"meteorological variable explanation":
                       ["Wind direction", "Wind speed Beaufort scale","flight rules"]})
explanation = st.checkbox('meteorologicalvariablecode explanation')
if explanation:
    st.table(df_expla)    
         
algorithms=[filename for filename in sorted(os.listdir("algo_list/")) if filename.endswith('.al')]

st.write("### **Select algorithm**")        
algorithm_file=st.selectbox("",(algorithms))
algo=pickle.load(open("algo_list/"+algorithm_file,"rb"))

   
#select quality report
st.sidebar.write("### **Select quality report**")
key_selected=st.sidebar.selectbox("",('Classification report','Confusion matrix','Precision','Recall','cros_val'))
reports={"Confusion matrix":0,"Precision":1,"Recall":2,"Classification report":3,"cros_val":0}


#defining url
today=pd.to_datetime("today")
head1="http://mandeo.meteogalicia.es/thredds/ncss/modelos/WRF_HIST/"+algo["mod_res"]
head2=today.strftime("/%Y/%m/wrf_arw_det_history_")+algo["mod_res"]
head3=today.strftime("_%Y%m%d_0000.nc4?")
head=head1+head2+head3
 
var1="var=dir&var=mod&var=wind_gust&var=mslp&var=temp&var=rh&var=visibility&var=lhflx"
var2="&var=lwflx&var=conv_prec&var=prec&var=swflx&var=shflx&var=cape&var=cin&var=cfh&var=T850"
var3="&var=cfl&var=cfm&var=cft&var=HGT500&var=HGT850&var=T500&var=snow_prec&var=snowlevel"
var=var1+var2+var3
 
f_day=(today+timedelta(days=2)).strftime("%Y-%m-%d") 
tail="&time_start="+today.strftime("%Y-%m-%d")+"T01%3A00%3A00Z&time_end="+f_day+"T23%3A00%3A00Z&accept=csv"
 

dffinal=pd.DataFrame() 
for coor in list(zip(algo["coor"][1:].lat.tolist(),algo["coor"][1:].lon.tolist(),np.arange(0,len(algo["coor"][1:].lat.tolist())).astype(str))):
  dffinal=pd.concat([dffinal,pd.read_csv(head+var+"&latitude="+str(coor[0])+"&longitude="+str(coor[1])+tail,).add_suffix(str(coor[2]))],axis=1)    
 
    
#filter all columns with lat lon and date
dffinal=dffinal.filter(regex='^(?!(lat|lon|date).*?)')
 
#remove column string between brakets
new_col=[c.split("[")[0]+c.split("]")[-1] for c in dffinal.columns]
for col in zip(dffinal.columns,new_col):
  dffinal=dffinal.rename(columns = {col[0]:col[1]})
 
dffinal=dffinal.set_index(pd.date_range(start=today.strftime("%Y-%m-%d"), end=(today+timedelta(days=3)).strftime("%Y-%m-%d"), freq="H")[1:-1]) 
 
#Machine learning and meteorological model probabilistic result 
 
#select x variable from the model
df_x=dffinal[algo["x_var"]]
 
#select D interval
if algo["D"]==0:
  df_x= df_x.iloc[0:23]
if algo["D"]==1:
  df_x=df_x.iloc[23:47]
if algo["D"]==2:
  df_x =df_x.iloc[47:72]
  
if algo["x_and_y_same"]:
    points_met_model=[e for e in algo["met_var_sc"].keys()]
    st.sidebar.write("### **Select meteorological model point**")
    point_met_model_selected=st.sidebar.slider("", min_value=0, max_value=len(points_met_model)-1, value=0)
    st.sidebar.title("""  Meteorological model report""")
    st.sidebar.dataframe(algo["met_var_sc"][points_met_model[point_met_model_selected]][reports[key_selected]])
    df_metmod_label=pd.cut(df_x[points_met_model[point_met_model_selected]],
                           bins=algo["interval"],).map({a:b for a,b in zip(algo["interval"],algo["labels"])})
     
  
 
#machine learning probabilistic result
ml_prob= algo["model"].predict_proba(algo["pca"].transform(algo["scaler"].transform(df_x))) 


ml_result=pd.DataFrame(ml_prob,columns=algo["model"].classes_).set_index(df_x.index).add_suffix('_ml')
ml_result["max_ml"]=ml_result.idxmax(axis=1)
ml_result.iloc[:,0:-1]=ml_result.iloc[:,0:-1].applymap(lambda n: '{:.0%}'.format(n))

#no columns with all 0%
ml_result_c=ml_result[ml_result != "0%"].dropna(axis=1, how='all').fillna("0%")

#get metar 

#today metar
OACI=''.join([c for c in algorithm_file if c.isupper()])

#url string
s1="https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py?station="
s2="&data=all"
s3="&year1="+today.strftime("%Y")+"&month1="+today.strftime("%m")+"&day1="+today.strftime("%d")
s4="&year2="+today.strftime("%Y")+"&month2="+today.strftime("%m")+"&day2="+today.strftime("%d")
s5="&tz=Etc%2FUTC&format=onlycomma&latlon=no&missing=M&trace=T&direct=no&report_type=1&report_type=2"
url=s1+OACI+s2+s3+s4+s5
df_metar_global=pd.read_csv(url,parse_dates=["valid"],).rename({"valid":"time"},axis=1)
df_metar=df_metar_global[["time","metar"]].set_index("time")

st.title(""" Deterministic  Forecast""")
if algo["x_and_y_same"]:
    compact_result=pd.concat([ml_result_c["max_ml"],df_metmod_label],axis=1).astype(str).join(df_metar,how="left")
    st.dataframe(compact_result)
    
else:
    st.dataframe(ml_result_c["max_ml"].to_frame().join(df_metar,how="left"))
    
st.title(""" Probabilistic Forecast""")   
st.dataframe(ml_result_c) 
    
st.sidebar.title("""  Machine learning report""")
st.sidebar.dataframe(algo[key_selected])

if st.checkbox('Show map and coordinates'):
    px.set_mapbox_access_token("pk.eyJ1IjoiZ3JhbmFudHVpbiIsImEiOiJja3B4dGU4OTkwMTFmMm9ycnNhMjJvaGJqIn0.VWzx_PkD9A5cSUVsn_ijCA")
    dist_map=px.scatter_mapbox(algo["coor"], hover_data=['distance'],lat='lat', lon='lon',color='distance',
                   color_continuous_scale=px.colors.cyclical.IceFire,)
    st.plotly_chart(dist_map)
    st.dataframe(algo["coor"])

if st.checkbox("Show abstract ?"):
    st.write(algo["abstract"])  
    
#"Download results ?"):
ml_result_c["time"] = ml_result_c.index
st.markdown(get_table_download_link(ml_result_c), unsafe_allow_html=True)
    
    
#select show x var , algorithm, pca and map
if st.sidebar.checkbox('Show meteorological model variables'):
    st.sidebar.write(algo["x_var"])
if st.sidebar.checkbox('Show machine learning algorithm'):
    st.sidebar.write(algo["model"])
if st.sidebar.checkbox('Show PCA algorithm'):
    st.sidebar.write(algo["pca"])
