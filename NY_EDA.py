
# coding: utf-8

# # Data Exploratory Analysis

# In[1]:


import csv
from pandas import ExcelWriter
import datetime, warnings  
import matplotlib as mpl
import numpy as np
import pandas as pd
import seaborn as sns
from seaborn import distplot
import sklearn as skl
import statistics
import statsmodels
import statsmodels.api as sm
from statsmodels.formula.api import ols

from scipy.stats import skew, kurtosis

from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy import stats
from sklearn import metrics, linear_model
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = 'all'


# In[2]:


city = pd.read_csv('C:/Users/Playdata/py_projects/Mentoring/historical-hourly-weather-data/city_attributes.csv')
humidity = pd.read_csv('C:/Users/Playdata/py_projects/Mentoring/historical-hourly-weather-data/humidity.csv')
pressure = pd.read_csv('C:/Users/Playdata/py_projects/Mentoring/historical-hourly-weather-data/pressure.csv')
temp = pd.read_csv('C:/Users/Playdata/py_projects/Mentoring/historical-hourly-weather-data/temperature.csv')
weath_desc = pd.read_csv('C:/Users/Playdata/py_projects/Mentoring/historical-hourly-weather-data/weather_description.csv')
windir= pd.read_csv('C:/Users/Playdata/py_projects/Mentoring/historical-hourly-weather-data/wind_direction.csv')
windspd=pd.read_csv('C:/Users/Playdata/py_projects/Mentoring/historical-hourly-weather-data/wind_speed.csv')


# In[83]:


add_weather = pd.read_csv('C:/Users/Playdata/py_projects/Mentoring/historical-hourly-weather-data/JFK_additional_weather.txt')


# In[86]:


add_weather.info()


# In[88]:





# # Basic Data Exploration

# In[3]:


file_list = [city, humidity, pressure, temp, weath_desc, windir, windspd]

for file in file_list:

    tab_info = pd.DataFrame(file.dtypes).T.rename(index={0:'column type'})
    tab_info = tab_info.append(pd.DataFrame(file.isnull().sum()).T.rename(index={0:'null values(nb)'}))
    tab_info=tab_info.append(pd.DataFrame(file.isnull().sum()/file.shape[0]*100).T.rename(index={0:'null values(%)'}))
    tab_info


# # Cleansing 

# ##  Weather.CSV

# ### Location(NY), YEAR(2015)

# In[4]:


city_ny = city[city['City']=='New York']
humidity_ny = humidity[['datetime','New York']]
pressure_ny = pressure[['datetime','New York']]
temp_ny = temp[['datetime','New York']]
weath_desc_ny = weath_desc[['datetime','New York']]
windir_ny = windir[['datetime','New York']]
windspd_ny = windspd[['datetime','New York']]


# In[5]:


def setyear(dataset):
    dataset = dataset[dataset['datetime'] >= '2014-12-31 22:00:00']
    dataset = dataset[dataset['datetime'] < '2016-01-01 02:00:00']
    #dataset = dataset
    return dataset


# In[6]:


datecolumn = humidity_ny[humidity_ny['datetime'] >= '2014-12-31 22:00:00']
datecolumn = datecolumn[humidity_ny['datetime'] < '2016-01-01 02:00:00']
datecolumn.drop('New York', axis = 1, inplace = True)
datecolumn.shape
datecolumn[:5]
datecolumn.tail()


# In[7]:


# changing data columns' name to each features, not location name
humidity_ny_date = setyear(humidity_ny)
humidity_ny_date_ren = humidity_ny_date.rename(columns = {'New York' : 'humidity'})

pressure_ny_date = setyear(pressure_ny)
pressure_ny_date_ren = pressure_ny_date.rename(columns = {'New York' : 'pressure'})

temp_ny_date = setyear(temp_ny)
temp_ny_date_ren = temp_ny_date.rename(columns = {'New York' : 'temperature'})

weath_desc_ny_date = setyear(weath_desc_ny)
weath_desc_ny_date_ren = weath_desc_ny_date.rename(columns = {'New York' : 'weather description'})

windir_ny_date = setyear(windir_ny)
windir_ny_date_ren = windir_ny_date.rename(columns = {'New York' : 'wind direction'})

windspd_ny_date = setyear(windspd_ny)
windspd_ny_date_ren = windspd_ny_date.rename(columns ={'New York' : 'wind speed'})


# In[8]:


weather = pd.DataFrame(columns = ['datetime', 'humidity', 'pressure', 'temperature', 'weather description', 'wind direction', 'wind speed'])
# , pressure_ny_date_ren, temp_ny_date_ren, weath_desc_ny_date_ren, windir_ny_date_ren, windspd_ny_date_ren)


# In[9]:


weather


# In[10]:


file_list_date = [city, humidity_ny_date, pressure_ny_date, temp_ny_date, weath_desc_ny_date, windir_ny_date, windspd_ny_date]
file_list_date_ren = [datecolumn['datetime'], humidity_ny_date_ren['humidity'], pressure_ny_date_ren['pressure'], temp_ny_date_ren['temperature'], weath_desc_ny_date_ren['weather description'], windir_ny_date_ren['wind direction'], windspd_ny_date_ren['wind speed']]
# for file in file_list_date_ren:
#     pd.concat()
# weather
weather = pd.concat(file_list_date_ren, axis = 1)
# # series로 변형
weather.to_csv('weather_transed.csv', sep='\t')

#파일 읽으려면 밑의 명령어와 같이 읽으면 됨.
# pd.read_csv('weather_transed.csv',sep='\t')


# In[11]:


weather[:5]
weather.tail()


# In[12]:


weather.shape
weather.info()


# ### Flight.csv

# In[13]:


df = pd.read_csv('C:/Users/Playdata/py_projects/Mentoring/input/flights.csv', low_memory=False)
print('Dataframe dimensions=', df.shape)


# In[14]:


df.info()
df.isnull().sum()


# In[15]:


df['DATE'] = pd.to_datetime(df[['YEAR','MONTH','DAY']])
df_2015_dep = df[df['ORIGIN_AIRPORT'] == "JFK"]
df_2015_arr = df[df['DESTINATION_AIRPORT'] =="JFK"]
df_2015 = df_2015_dep.append(df_2015_arr, ignore_index = True )


# In[16]:


#SCHEDULED_DEPARTURE = HHMM 의 int 형식으로 되어있음
# 시간으로 변환
def format_heure(chaine):
    if pd.isnull(chaine):
        return np.nan
    else:
        if chaine == 2400 : chaine = 0 
        chaine = "{0:04d}".format(int(chaine))
        heure = datetime.time(int(chaine[0:2]), int(chaine[2:4]))
        return heure
    
def combine_date_heure(x):
    if pd.isnull(x[0]) or pd.isnull(x[1]):
        return np.nan
    else:
        return datetime.datetime.combine(x[0],x[1])
    
def create_flight_time(df, col):    
    liste = []
    for index, cols in df[['DATE', col]].iterrows():    
        if pd.isnull(cols[1]):
            liste.append(np.nan)
        elif float(cols[1]) == 2400:
            cols[0] += datetime.timedelta(days=1)
            cols[1] = datetime.time(0,0)
            liste.append(combine_date_heure(cols))
        else:
            cols[1] = format_heure(cols[1])
            liste.append(combine_date_heure(cols))
    return pd.Series(liste)


# In[17]:


df_2015['SCHEDULED_DEPARTURE'] = create_flight_time(df_2015, 'SCHEDULED_DEPARTURE')
df_2015['DEPARTURE_TIME'] =df_2015['DEPARTURE_TIME'].apply(format_heure)
df_2015['SCHEDULED_ARRIVAL'] =df_2015['SCHEDULED_ARRIVAL'].apply(format_heure)
df_2015['ARRIVAL_TIME'] = df_2015['ARRIVAL_TIME'].apply(format_heure)

df_2015.loc[:5, ['SCHEDULED_DEPARTURE', 'SCHEDULED_ARRIVAL', 'DEPARTURE_TIME', 'ARRIVAL_TIME', 'DEPARTURE_DELAY', 'ARRIVAL_DELAY']]


# In[18]:


variables_to_remove = ['TAXI_OUT', 'TAXI_IN', 'WHEELS_ON', 'WHEELS_OFF','YEAR', 'MONTH','DAY','DAY_OF_WEEK','DATE', 'AIR_SYSTEM_DELAY','SECURITY_DELAY','AIRLINE_DELAY','LATE_AIRCRAFT_DELAY','WEATHER_DELAY', 'DIVERTED', 'CANCELLED','CANCELLATION_REASON','FLIGHT_NUMBER', 'TAIL_NUMBER', 'AIR_TIME']
df_2015.drop(variables_to_remove, axis = 1, inplace = True)
df_2015 = df_2015[['AIRLINE', 'ORIGIN_AIRPORT', 'DESTINATION_AIRPORT',
        'SCHEDULED_DEPARTURE', 'DEPARTURE_TIME', 'DEPARTURE_DELAY',
        'SCHEDULED_ARRIVAL', 'ARRIVAL_TIME', 'ARRIVAL_DELAY',
        'SCHEDULED_TIME', 'ELAPSED_TIME']]

df_2015[:5]


# In[19]:


df_2015 = df_2015[df_2015['ORIGIN_AIRPORT'] =='JFK']


# In[20]:


df_2015.to_csv('df_2015.csv', sep='\t')
# pd.read_csv('df_2015.csv', sep = "\t")


# In[21]:


df_2015.info()


# In[22]:


df_2015.isnull().sum()


# In[23]:


file_fin = [weather, df_2015]
for file in file_fin:
    tab_info = pd.DataFrame(file.dtypes).T.rename(index={0:'column type'})
    tab_info = tab_info.append(pd.DataFrame(file.isnull().sum()).T.rename(index={0:'null values(nb)'}))
    tab_info=tab_info.append(pd.DataFrame(file.isnull().sum()/df.shape[0]*100).T.rename(index={0:'null values(%)'}))
    tab_info

df_2015_wonull = pd.DataFrame.dropna(df_2015, how ='any')
df_2015_wonull.info()

file_fin = [weather, df_2015_wonull]
for file in file_fin:
    print(file.shape)
    tab_info = pd.DataFrame(file.dtypes).T.rename(index={0:'column type'})
    tab_info = tab_info.append(pd.DataFrame(file.isnull().sum()).T.rename(index={0:'null values(nb)'}))
    tab_info=tab_info.append(pd.DataFrame(file.isnull().sum()/df.shape[0]*100).T.rename(index={0:'null values(%)'}))

    tab_info


# In[24]:


weather.info()

df_2015_wonull.info()

def whether_delayed(time):
    if time >= 120:
        return 'y'
    else:
        return 'n'

df_2015_wonull['DEPARTURE DELAYED?'] = df_2015_wonull['DEPARTURE_DELAY'].apply(whether_delayed)

# df_2015_wonull
# df_2015_wonull.info()
df_fin = df_2015_wonull

df_fin[df_fin['DEPARTURE DELAYED?'] == 'y']


# In[25]:



## Normalisation

def normalize(dataset):
    dataSTNorm=((dataset-dataset.mean())/(dataset.max()-dataset.min()))*20
    dataNorm["diagnosis"]=dataset["diagnosis"]
    return dataNorm

weather_list_float = ['humidity', 'pressure', 'temperature', 'wind direction', 'wind speed']
humidity_std = statistics.stdev(weather['humidity'])
pressure_std = statistics.stdev(weather['pressure'])
temperature_std = statistics.stdev(weather['temperature'])
wind_direction_std = statistics.stdev(weather['wind direction'])
wind_speed_std = statistics.stdev(weather['wind speed'])

for i in weather_list_float:
    normed_list = weather[i].tolist()
    data_norm_list = []
    for norms in normed_list:
        data_norm = ((norms - np.mean(normed_list))/(np.max(normed_list) - np.min(normed_list)))
        data_norm_list.append(data_norm)
#     print(i, data_norm_list)
    weather[i+'_norm'] = data_norm_list

# 특정 시간에 관한 날씨 데이터
# 날씨 : a -> b
# 데이터 시간 : x시 -> y시 (1시간 차이라고 가정)
# 실제 비행편 x시 **분
# (a + |(b-a)| * (**/60))

weather['count'] = 1


# In[26]:


for i in weather_list_float:
    normed_list = weather[i].tolist()
    data_norm_list = []
    for norms in normed_list:
        data_norm = ((norms - np.mean(normed_list))/np.std(normed_list))
        data_norm_list.append(data_norm)
#     print(i, data_norm_list)
    weather[i+'_std'] = data_norm_list


# In[27]:


weather['datetime'] =  pd.to_datetime(weather['datetime'], infer_datetime_format =True)
df_fin['datetime'] = df_fin['SCHEDULED_DEPARTURE'].dt.round("H")


# In[28]:


# weather['datetime'] - pd.Timedelta(hours= 1)

# df_fin['datetime']

# dataset_fin_td1 = pd.merge(df_fin, weather, left_on = df_fin['datetime'], right_on = weather['datetime'] - pd.Timedelta(hours = 1) )
# dataset_fin1

# dataset_fin_td1


# In[29]:


df_fin['datetime']

dataset_fin = pd.merge(df_fin, weather, on='datetime')
dataset_fin

dataset_fin.info()
df_fin.info()


# In[30]:


dataset_fin.columns

dataset_final = dataset_fin[['AIRLINE', 'ORIGIN_AIRPORT', 'DESTINATION_AIRPORT', 'DEPARTURE DELAYED?','DEPARTURE_DELAY', 'datetime', 'weather description','humidity', 'humidity_std', 'pressure','pressure_std','temperature','temperature_std','wind direction', 'wind direction_std', 'wind speed', 'wind speed_std']]
dataset_final #91660 rows × 15 columns????????????


# In[31]:



df_fin

def delayiing(y):
    if y == 'y':
        return 1
    else:
        return 2
# dataset_final_dep_from_JFK['DEPARTURE_DELAY?'] = dataset_final_dep_from_JFK['DEPARTURE DELAYED?'].apply(delayiing)

dataset_final.info()


FINAL = pd.get_dummies(dataset_final, columns=['weather description'])

FINAL[:5]

# sns.pairplot(data = FINAL['weather description_scattered clouds', 'DEPARTURE DELAYED?'], hue = 'DEPARTURE DELAYED?')

FINAL_SET = FINAL[FINAL['ORIGIN_AIRPORT']=='JFK']

FINAL_SET.columns


# In[32]:


FINAL_SET['weather description'] = dataset_final['weather description']
FINAL_SET.columns


# In[33]:



FINAL_SET['DEPARTURE DELAYED?'] = FINAL_SET['DEPARTURE DELAYED?'].apply(delayiing)

FINAL_SET.shape #(91660, 40)

FINAL_SET.info()


# # Numerical and Graphical Univariate Analysis

# In[34]:


weather[:5]
weather.tail()

central tendency, spread, skewness, and kurtosis 

central tendency
    mean, median(robustnetss), mode
    
spread
    var, stdev
    ANOVA
    quantile, IQR, boxplot
    range

    Skewness (e) or kurtosis (u)            Conclusion
    −2SE(e) < e < 2SE(e)                    not skewed
    e ≤ −2SE(e)                            negative skew
    e ≥ 2SE(e)                             positive skew
    −2SE(u) < u < 2SE(u)                    not kurtotic
    u ≤ −2SE(u)                            negative kurtosis
    u ≥ 2SE(u)                             positive kurtosi

# In[35]:


weather.describe()


# In[36]:


humidity = weather['humidity']
pressure = weather['pressure']
temperature = weather['temperature']
wind_direction = weather['wind direction']
wind_speed = weather['wind speed']

columns = [humidity, pressure, temperature, wind_direction, wind_speed]


# In[37]:


humidity.describe()


# In[38]:


def basic_stat(x):
    print(x.name)
    print("skewness = ", skew(x))
    print("kurtosis = ", kurtosis(x))
    print(x.describe())
    print("median = ", np.median(x))
    print("mode = ", stats.mode(x))
    print('--------------------------------')
for col in columns:
    basic_stat(col)


# In[39]:


fig = plt.figure(1, figsize=(8,6))
gs = GridSpec(1,2)
ax1 = fig.add_subplot(gs[0,0])
ax2 = fig.add_subplot(gs[0,1])

ax2 = sns.distplot(weather['humidity'])
ax1.boxplot(weather['humidity'])


# In[40]:


fig = plt.figure(1, figsize=(8,6))
gs = GridSpec(1,2)
ax1 = fig.add_subplot(gs[0,0])
ax2 = fig.add_subplot(gs[0,1])
ax2 = sns.distplot(weather['pressure'])
ax1.boxplot(weather['pressure'])


# In[41]:


fig = plt.figure(1, figsize=(8,6))
gs = GridSpec(1,2)
ax1 = fig.add_subplot(gs[0,0])
ax2 = fig.add_subplot(gs[0,1])
ax2 = sns.distplot(weather['temperature'])
ax1.boxplot(weather['temperature'])


# In[42]:


fig = plt.figure(1, figsize=(8,6))
gs = GridSpec(1,2)
ax1 = fig.add_subplot(gs[0,0])
ax2 = fig.add_subplot(gs[0,1])
ax2 = sns.distplot(weather['wind direction'])
ax1.boxplot(weather['wind direction'])


# In[43]:


fig = plt.figure(1, figsize=(8,6))
gs = GridSpec(1,2)
ax1 = fig.add_subplot(gs[0,0])
ax2 = fig.add_subplot(gs[0,1])
ax2 = sns.distplot(weather['wind speed'])
ax1.boxplot(weather['wind speed'])


# In[44]:


def unique(list1): 
    unique_list = [] 
    for x in list1: 
        if x not in unique_list: 
            unique_list.append(x) 
    for x in unique_list: 
        print(x)


# In[45]:


unique(weather['weather description']) # 23, 원본 데이터는 38개


# In[46]:


fig = plt.figure(1, figsize=(8,6))
gs = GridSpec(1,2)
ax1 = fig.add_subplot(gs[0,0])
ax2 = fig.add_subplot(gs[0,1])

ax2 = sns.distplot(df_2015_wonull['DEPARTURE_DELAY'])
ax1.boxplot(df_2015_wonull['DEPARTURE_DELAY'])


# In[47]:


fig = plt.figure(1, figsize=(8,6))
gs = GridSpec(1,2)
ax1 = fig.add_subplot(gs[0,0])
ax2 = fig.add_subplot(gs[0,1])

ax2 = sns.distplot(df_2015_wonull['ARRIVAL_DELAY'])
ax1.boxplot(df_2015_wonull['ARRIVAL_DELAY'])


# In[48]:


fig = plt.figure(1, figsize=(8,6))
gs = GridSpec(1,2)
ax1 = fig.add_subplot(gs[0,0])
ax2 = fig.add_subplot(gs[0,1])

ax2 = sns.distplot(df_2015_wonull['SCHEDULED_TIME'])
ax1.boxplot(df_2015_wonull['SCHEDULED_TIME'])


# In[49]:


fig = plt.figure(1, figsize=(8,6))
gs = GridSpec(1,2)
ax1 = fig.add_subplot(gs[0,0])
ax2 = fig.add_subplot(gs[0,1])

ax2 = sns.distplot(df_2015_wonull['ELAPSED_TIME'])
ax1.boxplot(df_2015_wonull['ELAPSED_TIME'])


# In[50]:


a = sns.distplot(weather['humidity_std'])
a = sns.distplot(weather['pressure_std'])
a = sns.distplot(weather['temperature_std'])
a = sns.distplot(weather['wind direction_std'])
a = sns.distplot(weather['wind speed_std'])
a


# In[51]:


a = sns.distplot(weather['humidity_norm'])
a = sns.distplot(weather['pressure_norm'])
a = sns.distplot(weather['temperature_norm'])
a = sns.distplot(weather['wind direction_norm'])
a = sns.distplot(weather['wind speed_norm'])
a


# In[52]:


weather_pi= weather.pivot_table('count', 'weather description', 'humidity', aggfunc = 'sum')
sns.heatmap(weather_pi)


# In[53]:


weather[:5]
df_fin[:5]


# In[54]:


FINAL_SET['DEPARTURE_DELAY'].plot.hist()


# In[55]:


f, ax = plt.subplots(figsize=(10, 10))
sns.despine(f, left=True, bottom=True)
sns.scatterplot(x="temperature", y="wind speed",
                hue = "DEPARTURE DELAYED?",
                palette="ch:r=-.2,d=.3_r",
                size = "DEPARTURE_DELAY",
                sizes=(1,2),
                linewidth=0,
                data=FINAL_SET, ax=ax)


# In[56]:




f, ax = plt.subplots(figsize=(10, 10))
sns.despine(f, left=True, bottom=True)
sns.scatterplot(x="temperature", y="humidity",
                hue = "DEPARTURE DELAYED?",
                palette="ch:r=-.2,d=.3_r",
                sizes=(1, 8), linewidth=0,
                data=FINAL_SET, x_jitter = True, y_jitter = True, ax=ax)

f, ax = plt.subplots(figsize=(10, 10))
sns.despine(f, left=True, bottom=True)
sns.scatterplot(x="temperature", y="wind direction",
                hue = "DEPARTURE DELAYED?",
                palette="ch:r=-.2,d=.3_r",
                sizes=(1, 8), linewidth=0,
                data=FINAL_SET, x_jitter = True, y_jitter = True, ax=ax)

f, ax = plt.subplots(figsize=(10, 10))
sns.despine(f, left=True, bottom=True)
sns.scatterplot(x="temperature", y="pressure",
                hue = "DEPARTURE DELAYED?",
                palette="ch:r=-.2,d=.3_r",
                sizes=(1, 8), linewidth=0,
                data=FINAL_SET, x_jitter = True, y_jitter = True, ax=ax)


# In[57]:


# 'humidity', 'humidity_std',
#        'pressure', 'pressure_std', 'temperature', 'temperature_std',
#        'wind direction', 'wind direction_std', 'wind speed', 'wind speed_std',


# In[58]:


f, ax = plt.subplots(figsize=(10, 10))
sns.despine(f, left=True, bottom=True)
sns.scatterplot(x="temperature", y="pressure",
                hue = "DEPARTURE DELAYED?",
                palette="ch:r=-.2,d=.3_r",
                size = "DEPARTURE_DELAY",
                hue_order="DEPARTURE_DELAY",
                sizes=(1, 8), linewidth=0,
                data=FINAL_SET, x_jitter = True, y_jitter = True, ax=ax)


# In[59]:


FINAL_SET.columns


# In[60]:


# reviews[reviews['price'] < 100].plot.hexbin(x='price', y='points', gridsize=15)
FINAL_SET[FINAL_SET['DEPARTURE_DELAY'] >= 120].plot.hexbin(x='temperature', y="humidity", gridsize = 50)


# In[61]:


FINAL_SET[FINAL_SET['DEPARTURE_DELAY'] >= 120].plot.hexbin(x='temperature', y='pressure', gridsize = 15)
FINAL_SET[FINAL_SET['DEPARTURE_DELAY'] >= 120].plot.hexbin(x='temperature', y='wind direction', gridsize = 15)
FINAL_SET[FINAL_SET['DEPARTURE_DELAY'] >= 120].plot.hexbin(x='temperature', y='wind speed', gridsize = 15)
FINAL_SET[FINAL_SET['DEPARTURE_DELAY'] >= 120].plot.hexbin(x='humidity', y='pressure', gridsize = 15)
FINAL_SET[FINAL_SET['DEPARTURE_DELAY'] >= 120].plot.hexbin(x='humidity', y='wind direction', gridsize = 15)
FINAL_SET[FINAL_SET['DEPARTURE_DELAY'] >= 120].plot.hexbin(x='humidity', y='wind speed', gridsize = 15)
FINAL_SET[FINAL_SET['DEPARTURE_DELAY'] >= 120].plot.hexbin(x='pressure', y='wind direction', gridsize = 15)
FINAL_SET[FINAL_SET['DEPARTURE_DELAY'] >= 120].plot.hexbin(x='pressure', y='wind speed', gridsize = 15)


# In[62]:


FINAL_SET.plot.hexbin(x='humidity', y='DEPARTURE DELAYED?', gridsize = 15)
FINAL_SET.plot.hexbin(x='pressure', y='DEPARTURE DELAYED?', gridsize = 15)
FINAL_SET.plot.hexbin(x='temperature', y='DEPARTURE DELAYED?', gridsize = 15)
FINAL_SET.plot.hexbin(x='wind speed', y='DEPARTURE DELAYED?', gridsize = 15)
FINAL_SET.plot.hexbin(x='wind direction', y='DEPARTURE DELAYED?', gridsize = 15)


# In[63]:


import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from mpl_toolkits.mplot3d import Axes3D

#Parameters to set
mu_x = 0
variance_x = 3

mu_y = 0
variance_y = 15

#Create grid and multivariate normal
x = np.linspace(-10,10,500)
y = np.linspace(-10,10,500)
X, Y = np.meshgrid(x,y)
pos = np.empty(X.shape + (2,))
pos[:, :, 0] = X; pos[:, :, 1] = Y
rv = multivariate_normal([mu_x, mu_y], [[variance_x, 0], [0, variance_y]])

#Make a 3D plot
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot_surface(X, Y, rv.pdf(pos),cmap='viridis',linewidth=0)
ax.set_xlabel('X axis')
ax.set_ylabel('Y axis')
ax.set_zlabel('Z axis')
plt.show()


# In[64]:


def plott_3d(x,y):
    #Parameters to set
    mu_x = np.mean(x)
    variance_x = np.var(x)

    mu_y = np.mean(y)
    variance_y = np.var(y)
    
    
    min_x, min_y, max_x, max_y = np.min(x), np.min(y), np.max(x), np.max(y)
    
    #Create grid and multivariate normal
    x = np.linspace(min_x - 1, max_x +1, 500)
    y = np.linspace(0, 3)
    X, Y = np.meshgrid(x,y)
    pos = np.empty(X.shape + (2,))
    pos[:, :, 0] = X; pos[:, :, 1] = Y
    rv = multivariate_normal([mu_x, mu_y], [[variance_x, 0], [0, variance_y]])

    #Make a 3D plot
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot_surface(X, Y, rv.pdf(pos),cmap='viridis',linewidth=0)
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')
    plt.show()
    multivariate_normal()


# In[65]:


def plott_3d(x,y):
    #Parameters to set
    mu_x = np.mean(x)
    variance_x = np.var(x)

    mu_y = np.mean(y)
    variance_y = np.var(y)
    
    
    min_x, min_y, max_x, max_y = np.min(x), np.min(y), np.max(x), np.max(y)
    
    #Create grid and multivariate normal
    x = np.linspace(min_x - 1, max_x +1, 500)
    y = np.linspace(min_y - 1, max_y +1)
    X, Y = np.meshgrid(x,y)
    pos = np.empty(X.shape + (2,))
    pos[:, :, 0] = X; pos[:, :, 1] = Y
    rv = multivariate_normal([mu_x, mu_y], [[variance_x, 0], [0, variance_y]])

    #Make a 3D plot
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot_surface(X, Y, rv.pdf(pos),cmap='viridis',linewidth=0)
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')
    plt.show()
    multivariate_normal()
plott_3d(FINAL_SET['humidity'], FINAL_SET['DEPARTURE_DELAY'])
plott_3d(FINAL_SET['temperature'], FINAL_SET['DEPARTURE_DELAY'])
plott_3d(FINAL_SET['pressure'], FINAL_SET['DEPARTURE_DELAY'])
plott_3d(FINAL_SET['wind speed'], FINAL_SET['DEPARTURE_DELAY'])
plott_3d(FINAL_SET['wind direction'], FINAL_SET['DEPARTURE_DELAY'])


# In[66]:


plt.hist(FINAL_SET['DEPARTURE_DELAY'], color = 'blue', edgecolor = 'black', bins = 500)


# In[67]:


plt.hist(FINAL_SET['DEPARTURE_DELAY'], color = 'blue', range = (120,max(FINAL_SET['DEPARTURE_DELAY'])), edgecolor = 'black', bins = 500)


# In[68]:


FINAL_SET.columns


# 
# Descriptive statistics / univariate analysis
# 
# 
# central tendency, spread, skewness, and kurtosis 
# 
# central tendency
#     mean, median(robustnetss), mode
#     
# spread
#     var, stdev
#     ANOVA
#     quantile, IQR, boxplot
#     range
# 
#     Skewness (e) or kurtosis (u)            Conclusion
#     −2SE(e) < e < 2SE(e)                    not skewed
#     e ≤ −2SE(e)                            negative skew
#     e ≥ 2SE(e)                             positive skew
#     −2SE(u) < u < 2SE(u)                    not kurtotic
#     u ≤ −2SE(u)                            negative kurtosis
#     u ≥ 2SE(u)                             positive kurtosi
# 
# 
# bi/multivariate ananlysis
#     correlation analysis
#     quantitative feature - quantitative feature
#     categorical feature - quantitative feature
#     d
# 
# 
# 
# 
# 
# 

# In[69]:


FINAL_SET.hist(bins=100, figsize=(20,15))
plt.tight_layout()
plt.show()


# In[70]:


corr_matrix = FINAL_SET.corr()
corr_matrix


# In[71]:


plt.figure(figsize=(24, 20))

sns.heatmap(corr_matrix, 
            cmap='viridis', vmax=1.0, vmin=-1.0, linewidths=0.1,
            annot=True, annot_kws={"size": 8}, square=True);


# y값으로 1,2 범주형 변수가 아닌 departure delay 사용하고, hue는 departure delay가 120 이상인것에 표시(y,n)

# In[72]:


plt.figure(figsize=(24, 20))
gs = GridSpec(1,1)
ax1 = fig.add_subplot(gs[0,0])

ax1 = sns.countplot(x="weather description", hue="DEPARTURE DELAYED?", data=FINAL_SET)
ax1 = plt.tight_layout()
ax1 = plt.xticks(rotation=45)


# In[73]:


# plt.figure(figsize=(10,50))
# gs = GridSpec(1,1)
# ax1 = fig.add_subplot(gs[0,0])

# ax1 = sns.catplot(x="weather description", hue="DEPARTURE DELAYED?", col="DEPARTURE_DELAY", data=FINAL_SET, kind="count")
# ax1 = plt.tight_layout()
# ax1 = plt.xticks(rotation=45)


# In[74]:


FINAL_SET.columns


# In[75]:


# sns.pairplot(FINAL_SET,hue = 'DEPARTURE DELAYED?')


# In[76]:


FINAL_SET1 = FINAL_SET[['DEPARTURE DELAYED?', 'DEPARTURE_DELAY', 'datetime', 'humidity',
       'pressure', 'temperature', 'wind direction','wind speed']]
sns.pairplot(FINAL_SET1, hue = 'DEPARTURE DELAYED?')


# In[77]:


FINAL_SET['weather description'].astype('category')


# In[78]:


# FINAL_SET['datetime'] = pd.to_datetime(FINAL_SET['datetime'], infer_datetime_format = True)
FINAL_SET['datetime_hour'] = FINAL_SET['datetime'].dt.hour


# In[79]:


plt.figure(figsize=(24, 20))
gs = GridSpec(1,1)
ax1 = fig.add_subplot(gs[0,0])

ax1 = sns.countplot(x=FINAL_SET['datetime_hour'], hue="DEPARTURE DELAYED?", data=FINAL_SET)
ax1 = plt.tight_layout()
ax1 = plt.xticks(rotation=45)


# In[80]:


FINAL_SET1 = FINAL_SET[['DEPARTURE DELAYED?', 'DEPARTURE_DELAY', 'datetime_hour', 'humidity',
       'pressure', 'temperature', 'wind direction','wind speed']]
sns.pairplot(FINAL_SET1, hue = 'DEPARTURE DELAYED?')


# In[81]:


FINAL_SET['DESTINATION_AIRPORT'].astype('category')


# In[82]:


plt.figure(figsize=(24, 20))
gs = GridSpec(1,1)
ax1 = fig.add_subplot(gs[0,0])

ax1 = sns.countplot(x=FINAL_SET['DESTINATION_AIRPORT'], hue="DEPARTURE DELAYED?", data=FINAL_SET)
ax1 = plt.tight_layout()
ax1 = plt.xticks(rotation=45)

