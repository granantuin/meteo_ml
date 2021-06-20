
"""**Get database meteorological model and data from the station and select time periods**"""

#@title 
Database = "lat42.58lon-8.8046p4R4KmD0.csv" #@param ["lat42.225lon-8.63p10R4KmD0.csv", "lat42.225lon-8.63p10R4KmD1.csv", "lat42.225lon-8.63p10R4KmD2.csv", "lat42.58lon-8.8046p4R4KmD0.csv", "lat42.58lon-8.8046p4R4KmD1.csv", "lat42.626lon-8.7836p4R4KmD1.csv", "lat42.626lon-8.7836p4R4KmD0.csv", "lat42.626lon-8.7836p4R4KmD2.csv", "lat42.58lon-8.8046p4R4KmD2.csv"] {allow-input: true}
met_station = "coron.csv" #@param ["LEVX.csv", "coron.csv", "cortegada.csv"] {allow-input: true}
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


#get model variable independent or X and D lat lon too!!!
df_x=pd.read_csv("C:\\Users\\usuario\\Desktop\\colab\\ML_database\\wrf_database\\"+Database,parse_dates=["time"],).set_index("time")

#get observed variables or Y
df_y=pd.read_csv("C:\\Users\\usuario\\Desktop\\colab\\ML_database\\met_stations\\"+met_station,parse_dates=["time"],index_col="time")

#only data at the same time
df_all=pd.concat([df_x,df_y],axis=1).dropna()
#df_all=df_x[["mod0","mod1"]].join(df_y["metar_o"])

#select time
#df_all=df_all.query("index.dt.month in [8,]")
#df_all.query("index.dt.year in [2015] and mod0>2")
#df_all=df_all.query("index.dt.hour >=0 and index.dt.month in [12,1,2]")
df_all=df_all.drop(['Unnamed: 0'],axis=1)
print("independent variables :",df_x.columns[0:23].to_list());
print("dependent variables:",df_y.columns.to_list());
print(df_all.info(verbose=True));



"""**Classification problem**

**Defining independent variable and label Y**
"""

#defining independent variable and label y
y_var = ["prec_accumulated_1_hour_before_o"] #@param ["[\"mod_o\"]", "[\"temp_o\"]", "[\"spd_o\"]", "[\"visibility_o\"]", "[\"dir_o\"]", "[\"dir_o\",\"spd_o\"]", "[\"wind_gust_o\"]", "[\"wxcodes_o\"]", "[\"skyc1_o\"]", "[\"skyl1_o\"]", "[\"temp_o\"]", "[\"tempd_o\"]", "[\"gust_spd_o\"]", "[\"prec_accumulated_1_hour_before_o\"]"] {type:"raw", allow-input: true}

#raw Y variable y_var[0] for one variable
#df_all=df_all[df_all.dir_o!=-1]
Y_raw=df_all[y_var[0]]
df_l=pd.DataFrame()


"""
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

df_l[y_var[0]+"_l"]=pd.cut(round(Y_raw,0), bins=interval,retbins=False,labels=labels,precision=3)
df_l[y_var[0]+"_l"]=df_l[y_var[0]+"_l"].map({a:b for a,b in zip(interval,labels)})

"""
"""
#clouds height
Y_raw=pd.to_numeric(Y_raw, errors="coerce")*3.28084
#median 1500 feet
interval=pd.IntervalIndex.from_tuples([(-1, 1500),(1500,7000)])
labels=["<=1500ft",">1500ft"]
df_l[y_var[0]+"_l"]=pd.cut(Y_raw, bins=interval,retbins=False,labels=labels)
df_l[y_var[0]+"_l"]=df_l[y_var[0]+"_l"].map({a:b for a,b in zip(interval,labels)})
df_l[y_var[0]+"_l"]=df_l[y_var[0]+"_l"].astype(str).replace("nan","NC1")

"""

"""
#cloud cover.Warning!! VV codes empty spaces 
#print(df_all.skyc1_o.value_counts())

#df_all.loc[df_all.skyc1_o=="   ",["skyc1_o","metar_o"]].sample(30)
#df_all.loc[df_all.skyc1_o=="M",["skyc1_o","metar_o"]].sample(30)
df_all.loc[df_all.skyc1_o=="   ",["skyc1_o"]]="CAVOK"
df_all.loc[df_all.skyc1_o=="M",["skyc1_o"]]="CAVOK" 
df_all.loc[df_all.skyc1_o=="VV ",["skyc1_o"]]="VV" 
labels=['CAVOK', 'FEW', 'SCT', 'BKN', 'VV', 'NSC', 'OVC']
df_l[y_var[0]+"_l"]=df_all.skyc1_o

"""
"""
#rain or drizzle 1 something else 0
mask=df_all['wxcodes_o'].str.contains("RA")
df_all.loc[mask,["wxcodes_o"]]="1"
mask=df_all['wxcodes_o'].str.contains("DZ")
df_all.loc[mask,["wxcodes_o"]]="1"
df_all.loc[df_all.wxcodes_o!="1",["wxcodes_o"]]="0"
print(df_all.wxcodes_o.value_counts(normalize=True))
df_l[y_var[0]+"_l"]=df_all.wxcodes_o
labels=["0","1"]
interval=pd.IntervalIndex.from_tuples([(-1, 0),(0,500)])
"""
"""
#wind_gust
interval=pd.IntervalIndex.from_tuples([(0, 12.8611),(12.8611,500)])
labels=["<=25KT",">25KT"]
df_l[y_var[0]+"_l"]=pd.cut(pd.to_numeric(Y_raw,errors="coerce"), bins=interval,retbins=False,labels=labels)
df_l[y_var[0]+"_l"]=df_l[y_var[0]+"_l"].map({a:b for a,b in zip(interval,labels)})
#no gust "NG"
df_l[y_var[0]+"_l"]=df_l[y_var[0]+"_l"].astype(str).replace("nan","NG")

"""

"""
#metar wind intensity
interval=pd.IntervalIndex.from_tuples([(-1.5, 1.55),(1.55,2.60), (2.60, 4.20), (4.20, 7.72),
                                       (7.72,100)])
labels=['<=3KT', '(3KT-5KT]', '(5KT-8KT]', '(8KT-15KT]','>15KT']
df_l[y_var[0]+"_l"]=pd.cut(Y_raw, bins=interval,retbins=False,labels=labels)
df_l[y_var[0]+"_l"]=df_l[y_var[0]+"_l"].map({a:b for a,b in zip(interval,labels)})
"""
"""
#wind direction in metar VRB =-1
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
df_l[y_var[0]+"_l"]=pd.cut(Y_raw, bins=interval,retbins=False,labels=labels)
df_l[y_var[0]+"_l"]=df_l[y_var[0]+"_l"].map({a:b for a,b in zip(interval,labels)})

"""
"""

#define intervals and labels depending independent variable meteogalicia station
#two y_var dir_o and spd_o
threshold=2

interval=pd.IntervalIndex.from_tuples([(-1, 20), (20, 40), (40, 60),(60,80),
                                       (80,100),(100,120),(120,140),(140,160),
                                       (160,180),(180,200),(200,220),(220,240),
                                       (240,260),(260,280),(280,300),(300,320),
                                       (320,340),(340,360)])
labels=interval.astype(str)
df_l["dir_o"+"_l"]=pd.cut(Y_raw.iloc[:,0],bins=interval,retbins=False,precision=2,labels=labels).astype(str)
df_l["spd_o"]=Y_raw["spd_o"]
df_l.loc[df_l["spd_o"]<threshold,["dir_o_l"]]="VRB"

"""


"""

#wind direction
interval=pd.IntervalIndex.from_tuples([(-1, 90), (90, 180), (180, 270),(270,360)])
labels=["NE","SE","SW","NW"]
df_l[y_var[0]+"_l"]=pd.cut(Y_raw,bins=interval,retbins=False,precision=2,labels=labels)
df_l[y_var[0]+"_l"]=df_l[y_var[0]+"_l"].map({a:b for a,b in zip(interval,labels)})
"""

"""
#quantiles intervals
quant =  8
df_l[y_var[0]+"_l"]=pd.qcut(Y_raw, quant, retbins = False,precision=2,)
interval=pd.qcut(Y_raw, quant,retbins = True,precision=2)[0].cat.categories
"""

"""
#interval visibility in meters
interval=pd.IntervalIndex.from_tuples([(-1, 1000), (1000, 50000),])
labels=["<=1000 meters", "> 1000 meters"]
df_l[y_var[0]+"_l"]=pd.cut(Y_raw, bins=interval, retbins = False,precision=2,labels=labels)
df_l[y_var[0]+"_l"]=df_l[y_var[0]+"_l"].map({a:b for a,b in zip(interval,labels)})

"""
"""
#Beaufort wind intensity in m/s scale 
interval=pd.IntervalIndex.from_tuples([(-1, 0.5), (.5, 1.5), (1.5, 3.3),(3.3,5.5),
                                     (5.5,8),(8,10.7),(10.7,13.8),(13.8,17.1),
                                     (17.1,20.7),(20.7,24.4),(24.4,28.4),(28.4,32.6),(32.6,60)])
labels=["F0","F1","F2","F3","F4","F5","F6","F7","F8","F9","F10","F11","F12"]
df_l[y_var[0]+"_l"]=pd.cut(Y_raw, bins=interval, retbins = False,precision=2,labels=labels)
df_l[y_var[0]+"_l"]=df_l[y_var[0]+"_l"].map({a:b for a,b in zip(interval,labels)})

"""
"""
#wind direction metar report -1 variable wind "VRB"
interval=pd.IntervalIndex.from_tuples([(-2, -0.1), (-0.1, 90),(90,180),(180,270),(270,360)])
df_l[y_var+"_l"]=pd.cut(Y_raw, bins=interval, retbins = False,precision=2,)
"""

#prec accumulated hour before
interval=pd.IntervalIndex.from_tuples([(-1,0.1),(0.1,100)])
labels=["no rain","rain"]
df_l[y_var[0]+"_l"]=pd.cut(Y_raw, bins=interval, retbins = False,precision=2,labels=labels)
df_l[y_var[0]+"_l"]=df_l[y_var[0]+"_l"].map({a:b for a,b in zip(interval,labels)})
                                   
"""

#fr
df_l[y_var[0]+"_l"]=Y_raw
labels=["VFR","MVFR","IFR","LIFR"]

"""
#output Y labeled. Labels are category or string
Y=df_l[y_var[0]+"_l"]

#show results
df_all["Y_label"]=Y
df_all[["Y_label",y_var[0]]].sample(30)

"""**Defining independent variables. If X and Y are the same show meteorological model score**"""

#@title Define X variables .Warning!!! see manual variables
x_var = ["dir0","dir1","dir2","dir3"] #@param ["[\"mod0\",\"mod1\"]", "[\"mod0\",\"mod1\",\"mod2\",\"mod3\",\"mod4\",\"mod5\",\"mod6\",\"mod7\",\"mod8\",\"mod9\"]", "[\"temp0\",\"temp1\",\"temp2\"]", "[\"dir0\",\"dir1\",\"dir2\",\"dir3\",\"dir4\",\"dir5\",\"dir6\",\"dir7\",\"dir8\",\"dir9\"]", "['mod2', 'wind_gust2', 'mod0', 'wind_gust0', 'mod1', 'wind_gust1','mod3', 'wind_gust3']", "[\"dir0\",\"dir1\",\"dir2\",\"dir3\"]"] {type:"raw", allow-input: true}
x_and_y_same = True #@param {type:"boolean"}

from scipy.stats import entropy
from sklearn.metrics import classification_report

#add x manual
"""
x_var=["prec0","prec1","prec2","prec3","prec4","prec5","prec6","prec7","prec8","prec9"]


x_var=["visibility0","visibility1","visibility2","visibility3","visibility4",
       "visibility5","visibility6","visibility7","visibility8","visibility9"]


x_var=["wind_gust0","wind_gust1","wind_gust2","wind_gust3","wind_gust4","wind_gust5",
       "wind_gust6","wind_gust7","wind_gust8","wind_gust9"]


x_var=["rh0","rh1","rh3","rh4","rh5","cfl0","cfl1","cfl2","cfl3","cfl4","cft0","cft1",
       "cft2","cft3","cft4","visibility0","visibility2"]


x_var=["temp0","temp1","temp2","temp3","temp4","temp5"]

x_var=["wind_gust0","wind_gust1","wind_gust2","wind_gust3"]

x_var=["mod0","mod1","mod2","mod3","mod4","mod5","mod6","mod7","mod8","mod9"]

x_var=["prec0","prec1","prec2","prec3","conv_prec0","conv_prec1","conv_prec2","conv_prec3"]


x_var=["prec0","prec1","prec2","prec3"]

"""
x_var=["prec0","prec1","prec2","prec3"]



#transforunits       
X=df_all[x_var]


#shift

if x_and_y_same:
  met_var_sc={}
  for c in X.columns:
    print(c)
    sc_list=[]
    fig, axs = plt.subplots(3,figsize = (16,18))
    df_l[c+"_l"]=pd.cut(X[c],bins = interval,precision=2).astype(str)
    df_l[c+"_l"]=df_l[c+"_l"].map({a:b for a,b in zip(interval.astype(str),labels)})
    
    global_sc=pd.crosstab(df_l[y_var[0]+"_l"],df_l[c+"_l"], margins=True,)
    sc_list.append(global_sc)
    column_sc=pd.crosstab(df_l[y_var[0]+"_l"],df_l[c+"_l"], margins=True,normalize="columns")
    column_sc=column_sc.append(pd.DataFrame(entropy(column_sc,base=2),columns=["entropy"],
             index=column_sc.columns).T) 
    sc_list.append(column_sc)
    index_sc=pd.crosstab(df_l[y_var[0]+"_l"],df_l[c+"_l"], margins=True,normalize="index")
    sc_list.append(index_sc)
    clas_sc=pd.DataFrame(classification_report(df_l[y_var[0]+"_l"].astype(str),df_l[c+"_l"].astype(str),output_dict=True)).T
    sc_list.append(clas_sc)
    met_var_sc[c]=sc_list
    sns.heatmap(global_sc,annot=True,ax=axs[0],cmap="YlGnBu",fmt='.0f',)
    sns.heatmap(column_sc[:-1],annot=True,ax=axs[1],cmap="YlGnBu",fmt='.0%')
    sns.heatmap(index_sc,annot=True,ax=axs[2],cmap="YlGnBu",fmt=".0%")
    print(clas_sc)
    print("*************************************************************")

"""**sklearn version and update**"""

#@title sklearn version and update
import sklearn
print(sklearn.__version__)
#!pip install -U scikit-learn

"""**Machine learning algorithm from sklearn select x var and tune**"""

#@title Define variables .Warning!!! see manual variables


x_var = ["mod0","mod1","mod2","mod3","mod4","mod5","mod6","mod7","mod8","mod9"] #@param ["[\"mod0\",\"mod1\",\"mod2\",\"mod3\",\"mod4\",\"mod5\",\"mod6\",\"mod7\",\"mod8\",\"mod9\"]", "[\"cft0\",\"cfl0\",\"prec0\",\"lwflx0\",\"lhflx0\",\"visibility0\",\"cft1\",\"cfl1\",\"prec1\",\"lwflx1\",\"lhflx1\",\"visibility1\"]", "['mod2', 'wind_gust2', 'mod0', 'wind_gust0', 'mod1', 'wind_gust1','mod3', 'wind_gust3']"] {type:"raw", allow-input: true}

abstract = "Beaufort" #@param {type:"string"}
PCA_n =90#@param {type:"integer"}
"""
x_var=["cfl0","rh0","cft0","lwflx0","visibility0","cfl1","rh1","cft1"
,"lwflx1","visibility1","cfl2","rh2","cft2","lwflx2","visibility2","cfl3","rh3",
"cft3","lwflx3","visibility3","cfl4","rh4","cft4","lwflx4",
"visibility4","cfl5","rh5","cft5","lwflx5","visibility5","cfl6","rh6","cft6",
"lwflx6","visibility6"]

x_var=['mod2', 'wind_gust2', 'mod0', 'wind_gust0', 'mod1', 'wind_gust1','mod3',
       'wind_gust3',"dir0","dir1","dir2","dir3"]

x_var=["wind_gust0","wind_gust1","wind_gust2","wind_gust3","wind_gust4","wind_gust5",
       "wind_gust6","wind_gust7","wind_gust8","wind_gust9"]  

x_var=["visibility0","visibility3","rh4","rh7","cfl5","cfl8","cfl4"]

x_var=["prec0","prec1","prec2","prec3","prec4","prec5","prec6","prec7","HGT8500",
       "mslp0","cape0","cin0","cft0","dir0","HGT8501","rh0","mslp1","cape1",
       "rh1","cin1","cft1","dir1","mslp3","cape3","cin3","cft3","dir3","HGT8503"
       ,"rh3","mslp4","cape4","rh4","cin4","cft4","dir4"]

x_var=["rh0","rh1","rh3","rh4","rh5","cfl0","cfl1","cfl2","cfl3","cfl4","cft0","cft1",
       "cft2","cft3","cft4","visibility0","visibility2","visibility3","cape0",
       "prec0","prec1","prec2","prec3","cape2"]

x_var=["temp0","temp1","temp2","temp3","rh0","rh1","rh2","rh3"]


x_var=['cfl2', 'cft5', 'cft7', 'cft2', 'cfl5', 'cape6', 'cfl7', 'cft6', 'cfl6', 'cft8',
 'cape4', 'cfl8', 'cape5', 'cft9', 'cape1', 'cin6', 'rh3', 'cfl1', 'cfl9', 'cft3']

x_var=['mod2', 'wind_gust2', 'mod0', 'wind_gust0', 'mod1', 'wind_gust1','mod3',
       'wind_gust3',"dir0","dir1","dir2","dir3"]

x_var=["prec0","prec1","prec2","prec3"]

x_var=['prec3','prec0', 'prec1', 'cape0', 'visibility1', 'prec2']

x_var=["prec0","prec1","prec2","prec3"]

x_var=['cfl2', 'cfl8', 'cfl1', 'cfl0', 'prec9', 'lwflx3', 'prec0', 'rh3', 'cfl4',
            'cfl3', 'cape4', 'cfl9', 'dir0', 'cape5', 'cape6', 'T8503', 'rh2', 'prec3',
            'lwflx4', 'cfl6']

x_var=['dir0', 'mod0', 'wind_gust0', 'rh0', 'visibility0', 'lhflx0', 'lwflx0',
       'conv_prec0', 'prec0', 'swflx0', 'shflx0', 'cape0', 'cin0', 'cfh0',
       'cfl0', 'cfm0', 'cft0', 'dir1', 'mod1', 'wind_gust1', 'temp1', 'rh1',
       'visibility1', 'lhflx1', 'lwflx1', 'conv_prec1', 'prec1', 'swflx1',
       'shflx1', 'cape1', 'cin1', 'cfh1', 'cfl1', 'cfm1', 'cft1', 'HGT5001',
       'dir2', 'mod2', 'wind_gust2', 'rh2', 'visibility2', 'lhflx2', 'lwflx2',
       'conv_prec2', 'prec2', 'swflx2', 'shflx2', 'cape2', 'cin2', 'cfh2',
       'cfl2', 'cfm2', 'cft2', 'HGT8502', 'dir3', 'mod3', 'wind_gust3',
       'temp3', 'rh3', 'visibility3', 'lhflx3', 'lwflx3', 'conv_prec3',
       'prec3', 'swflx3', 'shflx3', 'cape3', 'cin3', 'cfh3', 'cfl3', 'cfm3',
       'cft3']


x_var=["prec0","prec1","prec2","prec3","prec4","prec5","prec6","prec7","HGT8500",
       "cape0","cin0","cft0","rh0","mslp1","cape1","rh1","cin1","cft1","dir1",
       "cape3","cin3","cft3","rh3","cape4","rh4","cin4","cft4","mod0","mod1"]

"""
x_var=['mod2', 'wind_gust2', 'mod0', 'wind_gust0', 'mod1', 'wind_gust1','mod3',
       'wind_gust3',"dir0","dir1","dir2","dir3"]
       
from sklearn.model_selection import RandomizedSearchCV
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import lightgbm as lgbm
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV 
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score,cross_validate
from sklearn.model_selection import ShuffleSplit
import pickle
import seaborn as sns
from scipy.stats import entropy


#Select X all or x_var
#X=df_all[x_var]
X=df_all[[ele for ele in df_all.columns if not ele.endswith("_o")][:-2]]


#split
x_train, x_test, y_train, y_test = train_test_split(X.values,Y.values,
                                                    test_size=0.1,
                                                    stratify=Y.values,
                                                    random_state=1)

#scaler X
scaler=StandardScaler().fit(x_train)
x_sc=scaler.transform(x_train)

#pca 
pca = PCA(n_components=PCA_n,svd_solver='arpack',random_state=1)
x_pca = pca.fit_transform(x_sc)

"""
#scaler X
scaler=StandardScaler()
x_sc=scaler.fit_transform(X)
#pca 
pca = PCA(n_components=PCA_n,svd_solver='auto',random_state=1)
x_pca = pca.fit_transform(x_sc)

#split
x_train, x_test, y_train, y_test = train_test_split(x_pca,Y.values, test_size=0.1,random_state=1)

"""


"""# model fit and results"""

#model = KNeighborsClassifier(n_neighbors=3)
#model = XGBClassifier(n_estimators=400,)
#model=RandomForestClassifier(n_estimators=50,class_weight='balanced')
#model=BalancedRandomForestClassifier(n_estimators=300)
#model=BalancedBaggingClassifier()
#model=BaggingClassifier()
#model=lgbm.LGBMClassifier()
#model=LogisticRegression()

model=MLPClassifier(hidden_layer_sizes=(300,100),learning_rate="adaptive",verbose=False, 
                    early_stopping=True,max_iter=2500,activation="logistic",alpha=0.0001, solver="adam")
#model=svm.SVC(kernel='rbf', class_weight={1: 7}, cache_size=1500, C=1,gamma=100)
#model=SGDClassifier(penalty="l1", loss= 'log', learning_rate= 'invscaling', eta0=100, class_weight= {1: 0.4, 0: 0.6}, alpha= 0.0001)
#model=DecisionTreeClassifier(random_state=1)
#model=LinearDiscriminantAnalysis(n_components=None, priors=None, shrinkage=0.7,solver='lsqr', store_covariance=False, tol=0.0001)


#model=ExtraTreesClassifier()


# Train the model using  pca
model.fit(x_pca,y_train)
y_pred=model.predict(pca.transform(scaler.transform(x_test)))


#y_pred=pycaret_model.predict(pd.DataFrame(pca.transform(scaler.transform(x_test))))
"""
# the model using x var
model.fit(x_train,y_train)
y_pred=model.predict(x_test)
"""


#Predict Output
global_ml=pd.crosstab(y_test,y_pred,margins=True,)
column_ml=pd.crosstab(y_test,y_pred,margins=True,normalize="columns")
column_ml=column_ml.append(pd.DataFrame(entropy(column_ml,base=2),columns=["entropy"],
             index=column_ml.columns).T)  
index_ml=pd.crosstab(y_test,y_pred, margins=True,normalize="index")

clas_ml=pd.DataFrame(classification_report(y_test,y_pred,output_dict=True)).T
fig, axs = plt.subplots(3,figsize = (12,14))
sns.heatmap(global_ml,annot=True,ax=axs[0],cmap="YlGnBu",fmt='.0f',)
sns.heatmap(column_ml[:-1],annot=True,ax=axs[1],cmap="YlGnBu",fmt='.0%')
sns.heatmap(index_ml,annot=True,ax=axs[2],cmap="YlGnBu",fmt=".0%")
print(clas_ml)

#get coordinates the n nearest points
df_r=pd.read_csv("C:\\Users\\usuario\\Desktop\\colab\\ML_database\\wrf_database\\"+"distan_"+Database[:-6]+".csv")

#create a dataframe with the coordinates (lat,lon, distance) dataframe first row station coordinates

df_coor=pd.concat([pd.DataFrame({"lat":[df_r["lat_st"][0]],"lon":[df_r["lon_st"][0]],
                                 "distance":[0]}),df_r]).reset_index(drop = True)[["lat","lon","distance"]]


#cross validation
cv = ShuffleSplit(n_splits=5, test_size=0.1, random_state=100)
cros_val=cross_validate(model, pca.transform(scaler.transform(X)), Y.values, cv=cv,scoring=["accuracy",'f1_macro',"f1_weighted"]) 
print("f1_weighted: %0.2f (+/- %0.2f)" % (cros_val['test_f1_weighted'].mean(), cros_val['test_f1_weighted'].std() * 2))
print("Accuracy: %0.2f (+/- %0.2f)" % (cros_val['test_accuracy'].mean(), cros_val['test_accuracy'].std() * 2))


#save scaler, pca and algorithm
if x_and_y_same:
  met_ml={"scaler":scaler,"pca":pca,"model":model,"Confusion matrix":global_ml,"Precision":column_ml,
        "Recall":index_ml,"Classification report":clas_ml,"met_var_sc":met_var_sc,"x_and_y_same":True,
        "abstract":abstract,"D":int(Database[-5:-4]),"interval":interval,"x_var":X.columns.to_list(),
        "y_var":y_var,"labels":labels,"cros_val":cros_val,"coor":df_coor,"mod_res":"d03"}
else:
   met_ml={"scaler":scaler,"pca":pca,"model":model,"Confusion matrix":global_ml,"Precision":column_ml,
        "Recall":index_ml,"Classification report":clas_ml,"x_and_y_same":False,"abstract":abstract,
        "D":int(Database[-5:-4]),"y_var":y_var,"x_var":X.columns.to_list(),
        "labels":labels,"cros_val":cros_val,"coor":df_coor,"mod_res":"d03"}

"""**Save algorithm**"""

file_prefix = "st_" #@param {type:"string"}
pickle.dump(met_ml, open("C:\\Users\\usuario\\Desktop\\colab\\algorithms\\"+file_prefix+y_var[0]+Database[:-3]+"al", 'wb'))



