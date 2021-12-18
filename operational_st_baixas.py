import numpy as np
import pandas as pd
from datetime import timedelta
import pickle
import streamlit as st
import plotly.express as px
import os
import base64
from st_aggrid import AgGrid
from io import BytesIO


def get_table_download_link(df):
    """
    Parameters
    ----------
    df : pandas Dataframe
    Returns
    -------
    Download xls file
    """
    output = BytesIO()
    writer = pd.ExcelWriter(output, engine='xlsxwriter')
    df.to_excel(writer, sheet_name='Sheet1')
    writer.save()
    val = output.getvalue()
    b64 = base64.b64encode(val)  # val looks like b'...'
    return f'<a href="data:application/octet-stream;base64,{b64.decode()}" download="report.xlsx">Download xls file</a>'


#set web page configuration
st.set_page_config(page_title="Meterological Machine Learning Platform",layout="wide")

# Select algorithm
st.write("#### **Select algorithm**")  
algorithms=[filename for filename in sorted(os.listdir("algo_list/")) if filename.endswith('.al')]

#algorithms explanation  
explanation = st.checkbox('meteorologicalvariable explanation')
if explanation:
    df_expla=pd.DataFrame({"meteorologicalvariablecode":["dir","spdb","fr"],"meteorological variable explanation":
                       ["Wind direction", "Wind speed Beaufort scale","flight rules"]})
    st.table(df_expla)    
algorithm_file=st.selectbox("meteorologicalvariable-[season]-meteorologicalstation-forecastday(version).al",(algorithms))
algo=pickle.load(open("algo_list/"+algorithm_file,"rb"))

#Show map
px.set_mapbox_access_token("pk.eyJ1IjoiZ3JhbmFudHVpbiIsImEiOiJja3B4dGU4OTkwMTFmMm9ycnNhMjJvaGJqIn0.VWzx_PkD9A5cSUVsn_ijCA")
dist_map=px.scatter_mapbox(algo["coor"], hover_data=['distance'],lat='lat', lon='lon',color='distance',
                           color_continuous_scale=px.colors.cyclical.IceFire,)
st.plotly_chart(dist_map)
st.markdown(get_table_download_link(algo["coor"]), unsafe_allow_html=True)


#Select model point if x and y are equal
if algo["x_and_y_same"]:
    points_met_model=[e for e in algo["met_var_sc"].keys()]
    #point_met_model_selected=st.slider("", min_value=0, max_value=len(points_met_model)-1, value=0)
    point_met_model_selected=st.radio("Select meteorological model point",np.arange(0,len(points_met_model)))
   
#select quality report
st.sidebar.write("#### **Select quality report**")
key_selected=st.sidebar.selectbox("",('Classification report','Confusion matrix','Precision','Recall','cros_val'))
reports={"Confusion matrix":0,"Precision":1,"Recall":2,"Classification report":3,"cros_val":0}


#defining url to get model from Meteogalicia server
today=pd.to_datetime("today")
#http://mandeo.meteogalicia.es/thredds/ncss/wrf_1km_baixas/fmrc/files/20211217/wrf_arw_det1km_history_d05_20211217_0000.nc4?
head1="http://mandeo.meteogalicia.es/thredds/ncss/wrf_1km_baixas/fmrc/files/"
head2=today.strftime("/%Y%m%d/wrf_arw_det1km_history_d05")
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
 
#select x variable from the model
df_x=dffinal[algo["x_var"]]
 
#select D interval
if algo["D"]==0:
  df_x= df_x.iloc[0:23]
if algo["D"]==1:
  df_x=df_x.iloc[23:47]
if algo["D"]==2:
  df_x =df_x.iloc[47:72]

 #meteorological model report 
if algo["x_and_y_same"]:    
    st.sidebar.title("""  Meteorological model report""")
    me_mo_re= algo["met_var_sc"][points_met_model[point_met_model_selected]][reports[key_selected]]
    st.sidebar.dataframe(me_mo_re)
    st.sidebar.markdown(get_table_download_link(me_mo_re), unsafe_allow_html=True)
    df_metmod_label=pd.cut(df_x[points_met_model[point_met_model_selected]],
                           bins=algo["interval"],).map({a:b for a,b in zip(algo["interval"],algo["labels"])})
     
#machine learning probabilistic result
ml_prob= algo["model"].predict_proba(algo["pca"].transform(algo["scaler"].transform(df_x))) 
ml_result=pd.DataFrame(ml_prob,columns=algo["model"].classes_).set_index(df_x.index).add_suffix('_ml')
ml_result["max_ml"]=ml_result.idxmax(axis=1)
ml_result.iloc[:,0:-1]=ml_result.iloc[:,0:-1].applymap(lambda n: '{:.0%}'.format(n))
#delete columns with all 0%
ml_result_c=ml_result[ml_result != "0%"].dropna(axis=1, how='all').fillna("0%")

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
    #st.dataframe(compact_result)
    AgGrid(compact_result.reset_index())
    st.markdown(get_table_download_link(compact_result), unsafe_allow_html=True)
    
else:
    compact_result= ml_result_c["max_ml"].to_frame().join(df_metar,how="left")
    #st.dataframe(compact_result)
    AgGrid(ml_result_c["max_ml"].to_frame().join(df_metar,how="left").reset_index())
    st.markdown(get_table_download_link(compact_result), unsafe_allow_html=True)
    
st.title(""" Probabilistic Forecast""")   
st.dataframe(ml_result_c) 
st.markdown(get_table_download_link(ml_result_c), unsafe_allow_html=True)
    
st.sidebar.title("""  Machine learning report""")
st.sidebar.dataframe(algo[key_selected])
st.sidebar.markdown(get_table_download_link(algo[key_selected]), unsafe_allow_html=True)

#select show x var , algorithm, pca and abstract
if st.sidebar.checkbox('Show meteorological model variables'):
    st.sidebar.write(algo["x_var"])
if st.sidebar.checkbox('Show machine learning algorithm'):
    st.sidebar.write(algo["model"])
if st.sidebar.checkbox('Show PCA algorithm'):
    st.sidebar.write(algo["pca"])
if st.sidebar.checkbox("Show abstract ?"):
    st.write(algo["abstract"])      
    