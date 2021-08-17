import os
import numpy as np
import pandas as pd

def select_metmodel_station():
    

    #select meteorogical model
    wrf_directory="C:\\Users\\usuario\\Desktop\\colab\\ML_database\\wrf_database\\"
    met_models=[filename for filename in os.listdir(wrf_directory) 
               if filename.endswith('.csv') and filename.startswith("l")]
    print("Select meteorological model number\n")
    for n,mod in zip(np.arange(0,len(met_models)),met_models):
        print(n,mod)
    wrf_number=int(input())    
    met_model=met_models[wrf_number]
    
    #select station
    stations_directory="C:\\Users\\usuario\\Desktop\\colab\\ML_database\\met_stations\\"
    station_list = [filename for filename in os.listdir(stations_directory) 
                    if filename.endswith('.csv')]
    print("Select station number\n")
    for n,mod in zip(np.arange(0,len(station_list)),station_list):
        print(n,mod)
    station_number=int(input())    
    met_station=station_list[station_number] 
    
    
        #get model variable independent or X and D lat lon too!!!
    df_x=pd.read_csv(wrf_directory+met_model,parse_dates=["time"],).set_index("time")
    
    #get observed variables or Y
    df_y=pd.read_csv(stations_directory+met_station,parse_dates=["time"],index_col="time")
    
    #only data at the same time
    df_all=pd.concat([df_x,df_y],axis=1).dropna()
    #df_all=df_x[["mod0","mod1"]].join(df_y["metar_o"])
    
    #select time
    #df_all=df_all.query("index.dt.month in [6,7,8]")
    #df_all.query("index.dt.year in [2015] and mod0>2")
    #df_all=df_all.query("index.dt.hour >=0 and index.dt.month in [12,1,2]")
    df_all=df_all.drop(['Unnamed: 0'],axis=1)
    print("independent variables :",df_x.columns[0:23].to_list());
    print("dependent variables:",df_y.columns.to_list());
    print(df_all.info(verbose=True));
    
    return met_model,met_station,df_x,df_y,df_all

def temp_o (Y_raw,y_var):   
        #temperatures dry and dew point
        interval=pd.IntervalIndex.from_tuples([(-243,271),(271, 273),(273,275),(275,277),(277,279),(279,281),
                                               (281,283),(283,285),(285,287),(287,289),(289,291),(291,293),
                                               (293,295),(295,297),(297,299),(299,301),(301,303),(303,305),(305,307)
                                               ,(307,309),(309,311),(311,313),(313,315),(315,317),(317,319),(319,321)
                                               ,(321,333)])
        
        labels=["(-30,-2]","(-2,0]","(0,2]","(2,4]","(4,6]","(6,8]","(8,10]","(10,12]",
                "(12,14]","(14,16]","(16,18]","(18,20]","(20,22]","(22,24]","(24,26]","(26,28]",
                "(28,30]","(30,32]","(32,34]","(34,36]","(36,38]","(38,40]","(40,42]",
                "(42,44]","(44,46]","(46,60]"]
        df_l=pd.DataFrame()
        df_l[y_var+"_l"]=pd.cut(round(Y_raw,0), bins=interval,retbins=False,labels=labels,precision=3)
        df_l[y_var+"_l"]=df_l[y_var+"_l"].map({a:b for a,b in zip(interval,labels)})
        Y=df_l[y_var+"_l"]
        return interval,labels,Y
    
def skyl1_o (Y_raw,y_var):
    
    #clouds height
    Y_raw=pd.to_numeric(Y_raw, errors="coerce")*3.28084
    #median 1500 feet
    interval=pd.IntervalIndex.from_tuples([(-1, 1500),(1500,7000)])
    labels=["<=1500ft",">1500ft"]
    df_l=pd.DataFrame()
    df_l[y_var+"_l"]=pd.cut(Y_raw, bins=interval,retbins=False,labels=labels)
    df_l[y_var+"_l"]=df_l[y_var+"_l"].map({a:b for a,b in zip(interval,labels)})
    df_l[y_var+"_l"]=df_l[y_var+"_l"].astype(str).replace("nan","NC1")
    Y=df_l[y_var+"_l"]
    return interval,labels,Y


def prec_metar_wx (Y_raw,y_var):
    
    
    df_l=pd.DataFrame()
    #rain or drizzle 1 something else 0
    mask=df_all['wxcodes_o'].str.contains("RA")
    df_all.loc[mask,["wxcodes_o"]]="precipitation"
    mask=df_all['wxcodes_o'].str.contains("DZ")
    df_all.loc[mask,["wxcodes_o"]]="precipitation"
    df_all.loc[df_all.wxcodes_o!="precipitation",["wxcodes_o"]]="no precipitation"
    print(df_all.wxcodes_o.value_counts(normalize=True))
    df_l[y_var+"_l"]=df_all.wxcodes_o
    labels=["no precipitation","precipitation"]
    interval=pd.IntervalIndex.from_tuples([(-1, 0.1),(0.1,500)])
    Y=df_l[y_var+"_l"]
    return interval,labels,Y
    
    


def skyc1_o (Y_raw, y_var):
    
    #cloud cover.Warning!! VV codes empty spaces 
    #print(df_all.skyc1_o.value_counts())
    
    #df_all.loc[df_all.skyc1_o=="   ",["skyc1_o","metar_o"]].sample(30)
    #df_all.loc[df_all.skyc1_o=="M",["skyc1_o","metar_o"]].sample(30)
    
    df_l=pd.DataFrame()
    df_all.loc[df_all.skyc1_o=="   ",["skyc1_o"]]="CAVOK"
    df_all.loc[df_all.skyc1_o=="M",["skyc1_o"]]="CAVOK" 
    df_all.loc[df_all.skyc1_o=="VV ",["skyc1_o"]]="VV" 
    labels=['CAVOK', 'FEW', 'SCT', 'BKN', 'VV', 'NSC', 'OVC']
    df_l[y_var+"_l"]=df_all.skyc1_o
    interval=[]
    Y=df_l[y_var+"_l"]
    return interval, labels, Y
    

def dir_o (Y_raw, y_var):
    
    #wind direction in metar VRB =-1
    df_l=pd.DataFrame()
    interval=pd.IntervalIndex.from_tuples([(-1.5, -0.5),(-0.5,20), (20, 40), (40, 60),
                                           (60,80),(80,100),(100,120),(120,140),(140,160),
                                           (160,180),(180,200),(200,220),(220,240),
                                           (240,260),(260,280),(280,300),(300,320),
                                           (320,340),(340,360)])
    labels=['VRB', '[0.0, 20.0]', '(20.0, 40.0]', '(40.0, 60.0]',
           '(60.0, 80.0]', '(80.0, 100.0]', '(100.0, 120.0]', '(120.0, 140.0]',
           '(140.0, 160.0]', '(160.0, 180.0]', '(180.0, 200.0]', '(200.0, 220.0]',
           '(220.0, 240.0]', '(240.0, 260.0]', '(260.0, 280.0]', '(280.0, 300.0]',
           '(300.0, 320.0]', '(320.0, 340.0]', '(340.0, 360.0]']
    df_l[y_var+"_l"]=pd.cut(Y_raw, bins=interval,retbins=False,labels=labels)
    df_l[y_var+"_l"]=df_l[y_var+"_l"].map({a:b for a,b in zip(interval,labels)})
    Y=df_l[y_var+"_l"]
    return interval, labels, Y
    
def spd_beaufort (Y_raw, y_var) : 
    
    
    #Beaufort wind intensity in m/s scale 
    df_l=pd.DataFrame()
    interval=pd.IntervalIndex.from_tuples([(-1, 0.5), (.5, 1.5), (1.5, 3.3),(3.3,5.5),
                                         (5.5,8),(8,10.7),(10.7,13.8),(13.8,17.1),
                                         (17.1,20.7),(20.7,24.4),(24.4,28.4),(28.4,32.6),(32.6,60)])
    labels=["F0","F1","F2","F3","F4","F5","F6","F7","F8","F9","F10","F11","F12"]
    df_l[y_var+"_l"]=pd.cut(Y_raw, bins=interval, retbins = False,precision=2,labels=labels)
    df_l[y_var+"_l"]=df_l[y_var+"_l"].map({a:b for a,b in zip(interval,labels)})
    Y=df_l[y_var+"_l"]
    return interval, labels, Y



def wind_gust (Y_raw, y_var): 
    #wind_gust
    df_l=pd.DataFrame()
    interval=pd.IntervalIndex.from_tuples([(0, 12.8611),(12.8611,500)])
    labels=["<=25KT",">25KT"]
    df_l[y_var+"_l"]=pd.cut(pd.to_numeric(Y_raw,errors="coerce"), bins=interval,retbins=False,labels=labels)
    df_l[y_var+"_l"]=df_l[y_var+"_l"].map({a:b for a,b in zip(interval,labels)})
    #no gust "NG"
    df_l[y_var+"_l"]=df_l[y_var+"_l"].astype(str).replace("nan","NG")
    Y=df_l[y_var+"_l"]
    return interval, labels, Y


def visibility (Y_raw, y_var): 
    
    df_l=pd.DataFrame()
    #interval visibility in meters
    interval=pd.IntervalIndex.from_tuples([(-1, 1000), (1000, 50000),])
    labels=["<=1000 meters", "> 1000 meters"]
    df_l[y_var+"_l"]=pd.cut(Y_raw, bins=interval, retbins = False,precision=2,labels=labels)
    df_l[y_var+"_l"]=df_l[y_var+"_l"].map({a:b for a,b in zip(interval,labels)})
    Y=df_l[y_var+"_l"]
    return interval, labels, Y



def prec_accumulated_1_hour_before (Y_raw, y_var): 
    
    df_l=pd.DataFrame()
    #prec accumulated hour before
    interval=pd.IntervalIndex.from_tuples([(-1,0.1),(0.1,100)])
    labels=["no rain","rain"]
    df_l[y_var+"_l"]=pd.cut(Y_raw, bins=interval, retbins = False,precision=2,labels=labels)
    df_l[y_var+"_l"]=df_l[y_var+"_l"].map({a:b for a,b in zip(interval,labels)})
    Y=df_l[y_var+"_l"]
    return interval, labels, Y

def spd_o_metar (Y_raw, y_var):
    df_l=pd.DataFrame()
    #metar wind intensity
    interval=pd.IntervalIndex.from_tuples([(-1.5, 1.55),(1.55,2.60), (2.60, 4.20), (4.20, 7.72),
                                           (7.72,100)])
    labels=['<=3KT', '(3KT-5KT]', '(5KT-8KT]', '(8KT-15KT]','>15KT']
    df_l[y_var+"_l"]=pd.cut(Y_raw, bins=interval,retbins=False,labels=labels)
    df_l[y_var+"_l"]=df_l[y_var+"_l"].map({a:b for a,b in zip(interval,labels)})
    Y=df_l[y_var+"_l"]
    return interval, labels, Y
    

def fr (Y_raw, y_var):
    df_l=pd.DataFrame()
    df_l[y_var+"_l"]  = Y_raw
    labels=["VFR","MVFR","IFR","LIFR"]
    interval=[]
    Y=df_l[y_var+"_l"]
    return interval, labels, Y


def label_y(df_all):
    
    
    """Defining independent variable and label Y """
    
    y_variables=[columns for columns in df_all.columns if columns.endswith("_o")]
    for n,mod in zip(np.arange(0,len(y_variables)),y_variables):
        print(n,mod)
    y_number=int(input("Select independent variable number\n"))    
    y_var=y_variables[y_number]
    
    Y_raw=df_all[y_var]
    
       
    
    if y_var=="skyl1_o":
       interval,labels,Y =skyl1_o (Y_raw, y_var)
    if y_var=="temp_o":
       interval,labels,Y =temp_o (Y_raw, y_var)
    if y_var=="skyc1_o":
       interval,labels,Y =skyc1_o (Y_raw, y_var)  
    if y_var=="wxcodes_o":
       interval,labels,Y = prec_metar_wx (Y_raw,y_var)  
    if y_var=="dir_o":
       interval,labels,Y = dir_o (Y_raw,y_var)  
    if y_var=="spd_o":
       functions=[spd_beaufort, spd_o_metar]
       for n,mod in zip(np.arange(0,len(functions)),functions):
           print(n,mod)
       f_number=int(input("Select label function number\n")) 
       interval,labels,Y =functions[f_number](Y_raw,y_var)    
         
    if y_var=="wind_gust_o":
       interval,labels,Y = wind_gust (Y_raw,y_var) 
    if y_var=="visibility_o":
       interval,labels,Y = visibility (Y_raw,y_var) 
    if y_var=="prec_accumulated_1_hour_before_o":
       interval,labels,Y = prec_accumulated_1_hour_before (Y_raw,y_var)   
    if y_var=="fr_o":
       interval,labels,Y = fr (Y_raw,y_var)   
    
    
    
    #show results
    df_all["Y_label"]=Y
    
    return interval , labels ,df_all 


met_model,met_station,df_x,df_y,df_all = select_metmodel_station()    
interval, labels, df_all = label_y(df_all)    