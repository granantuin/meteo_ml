import pandas as pd
import pickle
import streamlit as st


 

 
#load algorithm
algo=pickle.load(open("experimentalprec_accumulated_1_hour_before_olat42.58lon-8.8046p4R4KmD0.al","rb"))
st.write(algo["abstract"])
 

df_x=pd.read_csv("x_data.csv",index_col=0)
 
#machine learning probabilistic result
ml_prob= algo["model"].predict_proba(algo["pca"].transform(algo["scaler"].transform(df_x))) 
#ml_result=pd.DataFrame(ml_prob,columns=algo["model"].classes_).set_index(df_x.index).applymap("{:.0%}".format).add_suffix('_ml')
ml_result=pd.DataFrame(ml_prob,columns=algo["model"].classes_).set_index(df_x.index).add_suffix('_ml')
ml_result["max_ml"]=ml_result.idxmax(axis=1)
ml_result.iloc[:,0:-1]=ml_result.iloc[:,0:-1].applymap(lambda n: '{:.0%}'.format(n))
st.write("""  ## Machine learning results""")
st.write(ml_result)

