# -*- coding: utf-8 -*-
"""
Created on Fri Aug 27 09:28:43 2021

@author: usuario
"""
import os
import pandas as pd


metmodels=[filename for filename in sorted(os.listdir()) if filename.endswith('.csv') and not filename.startswith('d')]
print (metmodels)
df=pd.read_csv("lat42.58lon-8.8046p4R4KmD0.csv")
df[["mslp0","mslp1","mslp2","mslp3"]]=df[["mslp0","mslp1","mslp2","mslp3"]].div(100)
df.info()

df0=df.iloc[:,1:].astype("float32")
#df1=df.iloc[:,[1,7,24,30,47,53,70,76]].astype("int")


dft=pd.concat([df["time"],df0],axis=1)
dft=dft.drop(["conv_prec0","conv_prec1","conv_prec2","conv_prec3",
              "cft0","cft1","cft2","cft3",
              "lhflx0","lhflx1","lhflx2","lhflx3",
              "swflx0","swflx1","swflx2","swflx3",
              "lwflx0","lwflx1","lwflx2","lwflx3",
              "T5000","T5001","T5002","T5003"],axis=1)
dft.info()

dft.to_parquet("lat42.58lon-8.8046p4R4KmD0.parquet")