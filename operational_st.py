import numpy as np
import pandas as pd
from datetime import timedelta
import pickle
import streamlit as st

 
#load algorithm and select quality features

#☺st.image("Arousa.JPG")
st.sidebar.title('Choose your algorithm and quality feature')
algorithm_file=st.sidebar.selectbox('select algorithm',
                                    ("st_spd_olat42.58lon-8.8046p4R4KmD0.al",
                                     "st_spd_olat42.58lon-8.8046p4R4KmD0bis.al"))
algo=pickle.load(open(algorithm_file,"rb"))

   
#select quality report
key_selected=st.sidebar.selectbox("select quality report",('Confusion matrix', 'Precision', 'Recall', 'Classification report','cros_val'))
reports={"Confusion matrix":0,"Precision":1,"Recall":2,"Classification report":3,"cros_val":0}


#defining url
today=pd.to_datetime("today")
head1="http://mandeo.meteogalicia.es/thredds/ncss/modelos/WRF_HIST/"+algo["mod_res"]
head2=today.strftime("/%Y/%m/wrf_arw_det_history_")+algo["mod_res"]
head3=today.strftime("_%Y%m%d_0000.nc4?")
head=head1+head2+head3
 
var1="var=dir&var=mod&var=wind_gust&var=mslp&var=temp&var=rh&var=visibility&var=lhflx"
var2="&var=lwflx&var=conv_prec&var=prec&var=swflx&var=shflx&var=cape&var=cin&var=cfh"
var3="&var=cfl&var=cfm&var=cft&var=HGT500&var=HGT850&var=T500&var=T850"
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
    point_met_model_selected=st.sidebar.slider("select point", min_value=0, max_value=len(points_met_model)-1, value=0)
    st.sidebar.title("""  Meteorological model report""")
    st.sidebar.dataframe(algo["met_var_sc"][points_met_model[point_met_model_selected]][reports[key_selected]])
    st.title("Meteorological model forecast")
    df_metmod_label=pd.cut(df_x[points_met_model[point_met_model_selected]],
                           bins=algo["interval"],).map({a:b for a,b in zip(algo["interval"],algo["labels"])})
    st.dataframe(pd.DataFrame(df_metmod_label.values,columns=["p"+str(point_met_model_selected)]
                              ,index=df_metmod_label.index,dtype="string"))  
  
 
#machine learning probabilistic result
ml_prob= algo["model"].predict_proba(algo["pca"].transform(algo["scaler"].transform(df_x))) 


ml_result=pd.DataFrame(ml_prob,columns=algo["model"].classes_).set_index(df_x.index).add_suffix('_ml')
ml_result["max_ml"]=ml_result.idxmax(axis=1)
ml_result.iloc[:,0:-1]=ml_result.iloc[:,0:-1].applymap(lambda n: '{:.0%}'.format(n))

st.title("""   Machine learning forecast""")
st.dataframe(ml_result)

st.sidebar.title("""  Machine learning report""")
st.sidebar.dataframe(algo[key_selected])


#select show x var , algorithm, pca and map
if st.sidebar.checkbox('Show meteorological model variables'):
    st.sidebar.write(algo["x_var"])
if st.sidebar.checkbox('Show machine learning algorithm'):
    st.sidebar.write(algo["model"])
if st.sidebar.checkbox('Show PCA algorithm'):
    st.sidebar.write(algo["pca"])
if st.sidebar.checkbox('Show map and coordinates'):
    st.sidebar.map(algo["coor"])
    st.sidebar.dataframe(algo["coor"])
    

