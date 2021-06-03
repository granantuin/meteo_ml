import numpy as np
import pandas as pd
from datetime import timedelta
import pickle
import streamlit as st


 

 
#load algorithm
st.sidebar.image("Arousa.jpg")
st.sidebar.title('Choose your algorithm')
algorithm_file=st.sidebar.selectbox('select algorithm',('prec_1hour_before_coron_p4R4KmD0.al','B_spd_coron_p4R4KmD0.al'))
algo=pickle.load(open(algorithm_file,"rb"))
st.write(algo["abstract"])
 

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
for coor in list(zip(algo["lat"],algo["lon"],np.arange(0,len(algo["lat"])).astype(str))):
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
 
#machine learning probabilistic result
ml_prob= algo["model"].predict_proba(algo["pca"].transform(algo["scaler"].transform(df_x))) 


ml_result=pd.DataFrame(ml_prob,columns=algo["model"].classes_).set_index(df_x.index).add_suffix('_ml')
ml_result["max_ml"]=ml_result.idxmax(axis=1)
ml_result.iloc[:,0:-1]=ml_result.iloc[:,0:-1].applymap(lambda n: '{:.0%}'.format(n))
st.write("""  ## Machine learning results""")
st.dataframe(ml_result)





