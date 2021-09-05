

import pandas as pd



def select_metmodel_station():
    """ Select meteorological model and meteorological station """
    
    import numpy as np
    import os
    
    
    #select meteorogical model
    wrf_directory="C:\\Users\\usuario\\Desktop\\colab\\ML_database\\wrf_database\\"
    met_models=[filename for filename in os.listdir(wrf_directory) 
               if (filename.endswith('.csv') or filename.endswith('.pat')) and filename.startswith("l")]
    print("Select meteorological model number\n")
    for n,mod in zip(np.arange(0,len(met_models)),met_models):
        print(n,mod)
    wrf_number=int(input())    
    met_model=met_models[wrf_number]
    
    #select station
    stations_directory="C:\\Users\\usuario\\Desktop\\colab\\ML_database\\met_stations\\"
    station_list = [filename for filename in os.listdir(stations_directory) 
                    if filename.endswith('.csv') or filename.endswith('.pat')]
    print("Select station number\n")
    for n,mod in zip(np.arange(0,len(station_list)),station_list):
        print(n,mod)
    station_number=int(input())    
    met_station=station_list[station_number] 
    
    
    #get model variable independent or X and D lat lon too!!!
    # file csv or parquet
    if met_model.endswith( ".csv"):
        df_x=pd.read_csv(wrf_directory+met_model,index_col="time")
    else:
        df_x=pd.read_parquet(wrf_directory+met_model).set_index("time")
        
    #get observed variables or Y
    # file csv or parquet
    if met_station.endswith( ".csv"):
        df_y=pd.read_csv(stations_directory+met_station,index_col="time")
    else:
        df_y=pd.read_parquet(stations_directory+met_station).set_index("time")
        
    #only data at the same time
    df_all=pd.concat([df_x,df_y],axis=1).dropna()
    
    
    #select time
    #df_all=df_all.query("index.dt.month in [6,7,8]")
    #df_all.query("index.dt.year in [2015] and mod0>2")
    #df_all=df_all.query("index.dt.hour >=0 and index.dt.month in [12,1,2]")
    df_all=df_all.drop(['Unnamed: 0'],axis=1)
    print("independent variables :",df_x.columns[0:23].to_list());
    print("dependent variables:",df_y.columns.to_list());
    print(df_all.info(verbose=True));
    
    #get coordinates the n nearest points
    df_r=pd.read_csv("C:\\Users\\usuario\\Desktop\\colab\\ML_database\\wrf_database\\"+"distan_"+met_model[:-6]+".csv")

    #create a dataframe with the coordinates (lat,lon, distance) dataframe first row station coordinates

    df_coor=pd.concat([pd.DataFrame({"lat":[df_r["lat_st"][0]],"lon":[df_r["lon_st"][0]],
                                 "distance":[0]}),df_r]).reset_index(drop = True)[["lat","lon","distance"]]
    
    return met_model , met_station ,df_all ,df_coor

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


def label_y():
     
        
    """Defining independent variable and label Y """
    
    import numpy as np
    
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
    
    return interval , labels ,df_all , y_var


def metmodel_score (df_all,y_var):
    """check meteorological model """
    
    #import seaborn as sns
    #import matplotlib.pyplot as plt
    from scipy.stats import entropy
    from sklearn.metrics import classification_report
    
    x_and_y_same=input("met model same variables than independent (True or False)\n")
    if x_and_y_same == "True":
        print (df_all.columns [0:25])
        lstx=[]
        x_number=int(input("independent variables number\n"))
        for i in range(0, x_number):
            ele = str(input("variable number {}: \n".format(i)))
            lstx.append(ele)
        X=df_all[lstx] 
        met_var_sc={}
        for c in X.columns:
            df_l=pd.DataFrame()
            print(c)
            sc_list=[]
            
            df_l[c+"_l"]=pd.cut(X[c],bins = interval,precision=2).astype(str)
            df_l[c+"_l"]=df_l[c+"_l"].map({a:b for a,b in zip(interval.astype(str),labels)})
            df_l[y_var+"_l"]=df_all["Y_label"]
            global_sc=pd.crosstab(df_l[y_var+"_l"],df_l[c+"_l"], margins=True,)
            sc_list.append(global_sc)
            column_sc=pd.crosstab(df_l[y_var+"_l"],df_l[c+"_l"], margins=True,normalize="columns")
            column_sc=column_sc.append(pd.DataFrame(entropy(column_sc,base=2),columns=["entropy"],
                     index=column_sc.columns).T) 
            sc_list.append(column_sc)
            index_sc=pd.crosstab(df_l[y_var+"_l"],df_l[c+"_l"], margins=True,normalize="index")
            sc_list.append(index_sc)
            clas_sc=pd.DataFrame(classification_report(df_l[y_var+"_l"].astype(str),df_l[c+"_l"].astype(str),output_dict=True)).T
            sc_list.append(clas_sc)
            met_var_sc[c]=sc_list
            
            #fig, axs = plt.subplots(3,figsize = (16,18))
            #sns.heatmap(global_sc,annot=True,ax=axs[0],cmap="YlGnBu",fmt='.0f',)
            #sns.heatmap(column_sc[:-1],annot=True,ax=axs[1],cmap="YlGnBu",fmt='.0%')
            #sns.heatmap(index_sc,annot=True,ax=axs[2],cmap="YlGnBu",fmt=".0%")
            
            print(clas_sc)
            print("*************************************************************")
    else:
        met_var_sc={}
                
    return met_var_sc ,x_and_y_same
    
    
def selectx_pca_train(df_all):

    #@title sklearn version and update
    import sklearn
    print(sklearn.__version__)
    #!pip install -U scikit-learn
    
    
    
    
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    import lightgbm as lgbm
    from sklearn.ensemble import ExtraTreesClassifier
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.linear_model import SGDClassifier
    from sklearn.ensemble import BaggingClassifier
    from sklearn import svm
    from sklearn.model_selection import train_test_split
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.ensemble import RandomForestClassifier
    from xgboost import XGBClassifier
       
    
    
    print("model variables\n",[ele for ele in df_all.columns if not ele.endswith("_o")][:-2])
    all_x_var=input ("all x variables? (y/n)\n")
    if all_x_var=="y":
        X=df_all[[ele for ele in df_all.columns if not ele.endswith("_o")][:-2]]
        x_var=X.columns
    else:
        lsx=[]
        x_number=int(input("independent variables number\n"))
        for i in range(0, x_number):
            ele = str(input("variable number {}: \n".format(i)))
            lsx.append(ele)
        x_var=lsx
        X=df_all[x_var]
    #Select X all or x_var
    #X=df_all[x_var]
    
    PCA_n=int(input("PCA number less than {}\n".format(len(x_var)))) 
    
    #split better stratify=Y.values
    x_train, x_test, y_train, y_test = train_test_split(X.values,df_all.Y_label.values,
                                                        test_size=0.1,
                                                        #stratify=df_all.Y_label.values,
                                                        random_state=1)
    
    #scaler X
    scaler=StandardScaler().fit(x_train)
    x_sc=scaler.transform(x_train)
    
    #pca 
    pca = PCA(n_components=PCA_n,svd_solver='arpack',random_state=1)
    x_pca = pca.fit_transform(x_sc)
    
    #ml models 
    models=[KNeighborsClassifier(n_neighbors=3),  XGBClassifier(n_estimators=400),
            BaggingClassifier(),lgbm.LGBMClassifier(),LogisticRegression(),
            MLPClassifier(hidden_layer_sizes=(350,100),verbose=False,early_stopping=True,max_iter=2500,alpha=0.0001),
            svm.SVC(kernel='rbf', class_weight={1: 7}, cache_size=1500, C=1,gamma=100),
            SGDClassifier(eta0=100, class_weight= {1: 0.4, 0: 0.6}, alpha= 0.0001),
            DecisionTreeClassifier(random_state=1),
            LinearDiscriminantAnalysis(n_components=None, priors=None, shrinkage=0.7,solver='lsqr', store_covariance=False, tol=0.0001),       
            ExtraTreesClassifier(), RandomForestClassifier()]
    for i in range (0,len(models)):
        print (i ,": ", models[i])
    model_number= int(input ("Enter a model number\n"))
    model=models[model_number]
    
    # Train the model using  pca
    model.fit(x_pca,y_train)
    y_pred=model.predict(pca.transform(scaler.transform(x_test)))
    
    return x_var, scaler, pca, model, y_pred, y_test


def m_learning_sc (y_test, y_pred, x_var, scaler, pca, df_all):
    
    """ Machine learning score"""
    
    
    from scipy.stats import entropy
    from sklearn.metrics import classification_report
    #import matplotlib.pyplot as plt
    #import seaborn as sns
    
    global_ml=pd.crosstab(y_test,y_pred,margins=True,)
    column_ml=pd.crosstab(y_test,y_pred,margins=True,normalize="columns")
    column_ml=column_ml.append(pd.DataFrame(entropy(column_ml,base=2),columns=["entropy"],
                 index=column_ml.columns).T)  
    index_ml=pd.crosstab(y_test,y_pred, margins=True,normalize="index")
    
    clas_ml=pd.DataFrame(classification_report(y_test,y_pred,output_dict=True)).T
    
    #fig, axs = plt.subplots(3,figsize = (12,14))
    #sns.heatmap(global_ml,annot=True,ax=axs[0],cmap="YlGnBu",fmt='.0f',)
    #sns.heatmap(column_ml[:-1],annot=True,ax=axs[1],cmap="YlGnBu",fmt='.0%')
    #sns.heatmap(index_ml,annot=True,ax=axs[2],cmap="YlGnBu",fmt=".0%")
    print(clas_ml)
    
         
    
    
    return global_ml, column_ml, index_ml , clas_ml 


def save_al():
    
    #save scaler, pca and algorithm
    import pickle
    from sklearn.model_selection import ShuffleSplit
    from sklearn.model_selection import cross_validate
    
    print("cross validation. waiting...")
    cv = ShuffleSplit(n_splits=5, test_size=0.1, random_state=100)
    cros_val=cross_validate(model, pca.transform(scaler.transform(df_all[x_var])), df_all.Y_label, cv=cv,scoring=["accuracy",'f1_macro',"f1_weighted"]) 
    print("f1_weighted: %0.2f (+/- %0.2f)" % (cros_val['test_f1_weighted'].mean(), cros_val['test_f1_weighted'].std() * 2))
    print("Accuracy: %0.2f (+/- %0.2f)" % (cros_val['test_accuracy'].mean(), cros_val['test_accuracy'].std() * 2))
        
       
    """**Save algorithm**"""
    
    save_a=input("save algorithm (y/n)\n")
    if save_a=="y":
        
        abstract=str(input("abtract (sklearn version...)?\n"))
        if x_and_y_same=="True":
            
            met_ml={"scaler":scaler,"pca":pca,"model":model,"Confusion matrix":global_ml,"Precision":column_ml,
            "Recall":index_ml,"Classification report":clas_ml,"met_var_sc":met_var_sc,"x_and_y_same":True,
            "abstract":abstract,"D":int(met_model[-5:-4]),"interval":interval,"x_var":x_var,
            "y_var":y_var,"labels":labels,"cros_val":cros_val,"coor":df_coor,"mod_res":"d03"}
        else:
            
            met_ml={"scaler":scaler,"pca":pca,"model":model,"Confusion matrix":global_ml,"Precision":column_ml,
            "Recall":index_ml,"Classification report":clas_ml,"x_and_y_same":False,"abstract":abstract,
            "D":int(met_model[-5:-4]),"y_var":y_var,"x_var":x_var,
            "labels":labels,"cros_val":cros_val,"coor":df_coor,"mod_res":"d03"}
        
        file_name=input("algorithm filename (variable-station-(d0, d1...)?\n")
        pickle.dump(met_ml, open("C:\\Users\\usuario\\Desktop\\colab\\algorithms\\"+file_name+".al", 'wb'))



 
"""  *********** Main Program ************ """

met_model , met_station , df_all, df_coor = select_metmodel_station()    
interval, labels , df_all, y_var = label_y() 
met_var_sc , x_and_y_same = metmodel_score(df_all,y_var)

again="y"

while again=="y":
    x_var , scaler , pca , model , y_pred , y_test = selectx_pca_train(df_all)
    global_ml, column_ml, index_ml , clas_ml = m_learning_sc (y_test, y_pred, x_var, scaler, pca, df_all)
    again=input("repeat process ? (y/n) \n")
    
save_ml=input("Save ML model? (y/n)\n")   

if save_ml=="y":
    save_al()

   