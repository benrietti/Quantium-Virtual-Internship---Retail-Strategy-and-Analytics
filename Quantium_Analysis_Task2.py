#!/usr/bin/env python
# coding: utf-8

# Task: I am part of Quantium’s retail analytics team and have been approached by a client, the Category Manager for Chips, has asked us to test the impact of the new trial layouts with a data driven recommendation to whether or not the trial layout should be rolled out to all their stores.

# In[1]:


#imports
import numpy as np 
import pandas as pd

import datatable as dt

import os

import matplotlib 
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
plt.rc("font", size=14)
get_ipython().run_line_magic('matplotlib', 'inline')

import seaborn as sns

import scipy

import plotnine
import plotnine.data
from plotnine.data import economics

#ignore future warnings
import warnings 
warnings.filterwarnings('ignore') 


# In[2]:


#package/library versions
print("pandas version " + pd.__version__)
print("numpy version " + np.__version__)
print("scipy version " + scipy.__version__)
print("matplotlib version " + matplotlib.__version__)
print("seaborn version " + sns.__version__)
print("datatable" + dt.__version__)


# In[3]:


#load data into pandas Dataframe
df = pd.read_csv("QVI_data.csv")


# In[4]:


#view top of data
df.head()


# In[5]:


#evaluate datatypes and get the shape of the data
df.info()


# In[6]:


#check for duplicates
df.duplicated()


# In[7]:


#check for null values
print(df.isnull().sum())


# The client has selected store numbers 77, 86 and 88 as trial stores and want control stores to be established stores that are 
# 
# operational for the entire observation period.
# 
# Goal: Match trial stores to control stores that are similar to the trial store prior to the trial period of Feb 2019 in terms
# 
# of: 
# 
# - Monthly overall sales revenue
# - Monthly number of customers
# - Monthly number of transactions per customer
# 

# In[8]:


#create new column for month_id and convert values to yyyymm format and datatype to int64
df["DATE"] = pd.to_datetime(df["DATE"])
df["MONTH_ID"] = df["DATE"].dt.strftime("%Y%m").astype("int")


# In[9]:


#view changes
df.head()


# Next, I will define the measure calculations to use during the analysis. For each store and month I will calculate total sales, number of customers,transactions per customer, chips per customer and the average price per unit.

# In[10]:


#create 'price_per_unit' column in our data frame for ease of calculations
df['PRICE_PER_UNIT']=df['TOT_SALES']/df['PROD_QTY']


# In[11]:


#view changes
df.info()


# In[12]:


#classify values on the basis of "Store number" and "month ID" 
#group values according to store number and month ID
grp=df.groupby(['STORE_NBR','MONTH_ID'])
sumdf=grp.sum()
countdf = grp.nunique()

#create a result dataframe : 'resultdf' -- This dataframe contains the data required and it will further be used for cleaning
resultdf = grp.sum()[['TOT_SALES']]
resultdf['TOTAL_CUSTOMERS']=0
resultdf['TXN_PER_CUST']=0
resultdf['CHIPS_PER_CUST']=0
resultdf['AVG_PRICE_PER_UNIT']=0

#perform calculations 
totalcust = countdf['LYLTY_CARD_NBR']
txnpercust = countdf['TXN_ID']/countdf['LYLTY_CARD_NBR']
chipspercust = sumdf['PROD_QTY']/countdf['LYLTY_CARD_NBR']
avgpriceperunit = grp.mean()['PRICE_PER_UNIT']

#initialize values in dataframe
resultdf['TOTAL_CUSTOMERS']+=totalcust
resultdf['TXN_PER_CUST']+=txnpercust
resultdf['CHIPS_PER_CUST']+=chipspercust
resultdf['AVG_PRICE_PER_UNIT']+=avgpriceperunit
resultdf


# In[13]:


#create a data frame "allTimeData" -- containing data of stores that are available for whole observation period
#remove the stores which are not available for full observation periods 
allTimeData = resultdf.copy()
def check12():
    for i in range(272):
        if allTimeData.loc[i+1].index.nunique() != 12:
            #print("{} : {}".format(i+1,allTimeData.loc[i+1].index.nunique())) #Uncomment to find which Stores are removed
            allTimeData.drop(i+1,inplace=True)
check12()
allTimeData.info()


# In[14]:


#create a data frame that contains all the values before beginning of trial period
#consider data before trial period begins
pretrialdf= allTimeData.copy()
pretrialdf.reset_index(inplace=True)
pretrialdf = pretrialdf[pretrialdf['MONTH_ID']<201902]
pretrialdf.set_index(['STORE_NBR','MONTH_ID'],inplace=True)
pretrialdf.info()


# In[15]:


#create a dataframe that contains only the data of trial stores (i.e. 77, 86, 88)
trialStoresdf = pretrialdf.loc[[77,86,88]]
#remove the trial stores from the data to avoid selecting the trial store itself
pretrialdf.drop([86,88,77],inplace=True)


# In[16]:


#find correlation with other stores
row_indexes = list(pretrialdf.xs(201807,level='MONTH_ID').index)
yearMonths = list(pretrialdf.loc[1].index)
corrTable=pd.DataFrame(index=row_indexes)
corrTable['nSales']=0
corrTable['nCustomers']=0
corrTable.index.names=['STORE_NBR']

#function to calculate correlation
def calcCorr(trialStoreNo):
    corrT = corrTable.copy()
    trialtmp = trialStoresdf.copy()
    pretmp = pretrialdf.copy()    
    seriestmp = trialtmp.loc[trialStoreNo]
    s_nSales = seriestmp['TOT_SALES']
    s_nCustomers = seriestmp['TOTAL_CUSTOMERS']
    for i in row_indexes:
        df1 = pretmp.loc[i]
        d_nSales = df1[['TOT_SALES']]
        d_nCustomers = df1[['TOTAL_CUSTOMERS']]
        corr_nSales = d_nSales.corrwith(s_nSales)
        corr_nCustomers = d_nCustomers.corrwith(s_nCustomers)
        corrT['nSales'].loc[i] = corr_nSales['TOT_SALES'] 
        corrT['nCustomers'].loc[i]=corr_nCustomers['TOTAL_CUSTOMERS']
    return corrT


# In[17]:


#calculate correlation with trial store-77
corrSt_77 = calcCorr(77)
#calculate correlation with trial store-86
corrSt_86 = calcCorr(86)
#calculate correlation with trial store-88
corrSt_88 = calcCorr(88)
#print("\nCorrealtion with Store 77 :- \n\n{}".format(corrSt_77.head()))
#print("\nCorrealtion with Store 86 :- \n\n{}".format(corrSt_86.head()))
#print("\nCorrealtion with Store 88 :- \n\n{}".format(corrSt_88.head()))


# In[18]:


#find "standardized magnitude distance" with other stores 
#find magnitude of distance
def magnDist(trialSt):
        tmpdataframe = trialStoresdf.loc[trialSt].copy()
        tmpdataframe = abs(pretrialdf-tmpdataframe)
        #To standardise the magnitude distance
        maxseries = tmpdataframe.max()
        minseries = tmpdataframe.min()
        tmpdataframe = 1 - ((tmpdataframe-minseries)/(maxseries-minseries))
        tmpdf = tmpdataframe.reset_index().groupby('STORE_NBR').mean().drop('MONTH_ID',axis=1)[['TOT_SALES','TOTAL_CUSTOMERS']]
        return tmpdf

#find std magnitude distance between trial and other stores
magnDist77 = magnDist(77)
magnDist86 = magnDist(86)
magnDist88 = magnDist(88)
#print("\nStd distance with Store 77 :- \n\n{}".format(magnDist77.head()))
#print("\nStd distance with Store 86 :- \n\n{}".format(magnDist86.head()))
#print("\nStd distance with Store 88 :- \n\n{}".format(magnDist88.head()))


# In[19]:


#find CONTROL center
#calculate total score and find control center
def calcTotalScore(corrTbl,magnTbl):
    combinedScoreTable = 0
    #After merging all 4 scores
    combinedScoreTable = pd.concat([corrTbl,magnTbl],axis=1)
    #Calculating total score
    combinedScoreTable['Total_Score']=combinedScoreTable.mean(axis=1)
    #Maximum score
    storeNBR = combinedScoreTable[combinedScoreTable['Total_Score']==combinedScoreTable['Total_Score'].max()].index[0]
    return storeNBR


# In[20]:


#fetch sum() value of a particular column of a store before the beginning of trial period
def fetchVal(store_n,col_name):
    i = allTimeData.xs(store_n,level='STORE_NBR').loc[:201901].sum()[col_name]
    i=round(i,2)
    return i


# In[21]:


#calculate scaling factor
def calcScalingFactor(trial_n,control_n,col_name):
    trial_nValue = fetchVal(trial_n,col_name)
    control_nValue = fetchVal(control_n,col_name)
    return trial_nValue/control_nValue


# In[22]:


#Store-77 : Performance Analysis
store77 = calcTotalScore(corrSt_77,magnDist77)
print("Control store for Store 77 is Store {}".format(store77))


# In[23]:


#Setting variables for ease of code
TrialStoreNo = 77
ControlStoreNo = store77


# In[24]:


#comparison table for total sales and customers
control_Sales = fetchVal(ControlStoreNo,'TOT_SALES')
control_Cust = fetchVal(ControlStoreNo ,'TOTAL_CUSTOMERS')
trial_Sales = fetchVal(TrialStoreNo,'TOT_SALES')
trial_Cust = fetchVal(TrialStoreNo,'TOTAL_CUSTOMERS')

#create new dataframe 'compartable'
comparTable = pd.DataFrame(index=['TOT_SALES','TOT_CUSTOMER'],columns=['TRIAL_ST','CONTROL_ST'])
comparTable['TRIAL_ST']['TOT_SALES'] = trial_Sales
comparTable['TRIAL_ST']['TOT_CUSTOMER'] = trial_Cust
comparTable['CONTROL_ST']['TOT_SALES'] = control_Sales
comparTable['CONTROL_ST']['TOT_CUSTOMER'] = control_Cust
comparTable=comparTable.transpose()


# In[25]:


#compare data of trial store and control store
x=comparTable
fig =plt.figure(figsize=(9,2.5),dpi=100)
ax = fig.add_axes([0,0,1,1])
plt.subplot(1,2,1)
plt.title("Total Sales comparison")
sns.barplot(x=x.index,y='TOT_SALES',data=x)
plt.subplot(1,2,2)
plt.title("Total Customers comparison")
sns.barplot(x=x.index,y='TOT_CUSTOMER',data=x)
plt.tight_layout()


# In[126]:


#Sales of store-77
#assign Variables for ease of code
columnName = 'TOT_SALES'


# In[27]:


#scaling
scalingFactor = calcScalingFactor(TrialStoreNo,ControlStoreNo,columnName)
scalingFactor


# In[28]:


#make dataframes needed for Control & Trial Center
controlStoredata = allTimeData.loc[ControlStoreNo].copy()
trialstoredata = allTimeData.loc[TrialStoreNo].copy()


# In[29]:


#applying scaling factor
#after scaling
scaleddata=controlStoredata[[columnName]]*scalingFactor
scaleddata.head()


# In[128]:


#percent diff
percentDiff_Sales = abs(scaleddata[[columnName]]-trialstoredata[[columnName]])*100/scaleddata[[columnName]]
percentDiff_Sales.head()


# In[31]:


#mean/std. deviation
meanControl = scaleddata.loc[201902:201904].copy()
m_Cont = meanControl[columnName].mean()
m_Cont = round(m_Cont,2)

meanTrial = trialstoredata.loc[201902:201904].copy()
m_Trial = meanTrial[columnName].mean()
m_Trial = round(m_Trial,2)

stdDev = (percentDiff_Sales.loc[:201901]).std()
stdDev = stdDev[columnName]
stdDev = round(stdDev,4)

print("Mean of scaled control store = {}".format(m_Cont))
print("Mean of Trial store = {}".format(m_Trial))
print("Standard deviation of data = {}".format(stdDev))


# In[32]:


#perform t-test for alpha=0.05
from scipy.stats import t
#calculate t-statistic
t_stat = abs(m_Cont-m_Trial)/stdDev
t_stat = round(t_stat,4)
#calculate critical t-value
alpha=0.05
degree = 8-1
cv_t = t.ppf(1-alpha, degree)
cv_t = round(cv_t,4)
print("t-statistic = {}".format(t_stat))
print("t-critical = {}".format(cv_t))


# In[33]:


#visualize the data
newName = "Total_Sales"
#data of CONTROL STORE
tmp_Controldf = controlStoredata.reset_index()
tmp_Controldf = tmp_Controldf[['MONTH_ID',columnName]]
tmp_Controldf['Store_Type']="Control_Store"
tmp_Controldf.rename(columns={'MONTH_ID':"Transaction_Month",columnName:newName},inplace=True)
tmp_Controldf.head()


# In[34]:


#data of TRIAL STORE
tmp_Trialdf = trialstoredata.reset_index()
tmp_Trialdf = tmp_Trialdf[['MONTH_ID',columnName]]
tmp_Trialdf['Store_Type']="Trial_Store"
tmp_Trialdf.rename(columns={'MONTH_ID':"Transaction_Month",columnName:newName},inplace=True)
tmp_Trialdf.head()


# In[35]:


#data of CONTROL store with 95% confidence
pastSales_Control05 = tmp_Controldf.copy()
pastSales_Control05[newName] = pastSales_Control05[newName]  - (stdDev*2)
pastSales_Control05['Store_Type'] = "Control 5 % Confidence"
pastSales_Control05.head()


# In[36]:


#data of CONTROL store with 5% confidence
pastSales_Control95 = tmp_Controldf.copy()
pastSales_Control95[newName] = pastSales_Control95[newName]  + (stdDev*2)
pastSales_Control95['Store_Type'] = "Control 95 % Confidence"
pastSales_Control95.head()


# In[37]:


#merge all data into one dataframe
pastSales = pd.DataFrame(columns=['Store_Type',newName,'Transaction_Month'])
pastSales = pd.concat([pastSales,pastSales_Control05,tmp_Trialdf,tmp_Controldf,pastSales_Control95])
pastSales.reset_index(inplace = True)
pastSales.drop('index',axis=1,inplace=True)
pastSales.info()


# In[38]:


#view data
trialPeriodSales = pastSales[(pastSales['Transaction_Month']<201905) & (pastSales['Transaction_Month']>201901)]
trialPeriodSales


# In[39]:


#delete temporary dataframes
del pastSales_Control05
del pastSales_Control95
del tmp_Controldf
del tmp_Trialdf


# In[40]:


#view data
fig =plt.figure(figsize=(8,3),dpi=100)
ax = fig.add_axes([0,0,1,1])
sns.barplot(y=newName,x='Transaction_Month',hue='Store_Type',data=trialPeriodSales)
plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))


# Observation: the trial in store 77 is significantly different to its control store in the trial period ; This is because the trial store performance lies outside the 5% to 95% confidence interval of the control store in two of the three trial months
# 
# 

# In[127]:


#Customers of Store-77
columnName = 'TOTAL_CUSTOMERS'


# In[42]:


#scaling
scalingFactor = calcScalingFactor(TrialStoreNo,ControlStoreNo,columnName)
scalingFactor


# In[43]:


#make data frames to be used later
controlStoredata = allTimeData.loc[ControlStoreNo].copy()
trialstoredata = allTimeData.loc[TrialStoreNo].copy()


# In[44]:


#after scaling
scaleddata=controlStoredata[[columnName]]*scalingFactor
scaleddata.head()


# In[45]:


#percent diff
percentDiff_Sales = abs(scaleddata[[columnName]]-trialstoredata[[columnName]])*100/scaleddata[[columnName]]
percentDiff_Sales.head()


# In[46]:


#mean/std. deviation
meanControl = scaleddata.loc[201902:201904].copy()
m_Cont = meanControl[columnName].mean()
m_Cont = round(m_Cont,2)

meanTrial = trialstoredata.loc[201902:201904].copy()
m_Trial = meanTrial[columnName].mean()
m_Trial = round(m_Trial,2)

stdDev = (percentDiff_Sales.loc[:201901]).std()
stdDev = stdDev[columnName]
stdDev = round(stdDev,4)

print("Mean of scaled control store = {}".format(m_Cont))
print("Mean of Trial store = {}".format(m_Trial))
print("Standard deviation of data = {}".format(stdDev))


# In[47]:


#perform t-test for alpha=0.05
from scipy.stats import t
#calculate t-statistic
t_stat = abs(m_Cont-m_Trial)/stdDev
t_stat = round(t_stat,4)
#calculate critical t-value
alpha=0.05
degree = 8-1
cv_t = t.ppf(1-alpha, degree)
cv_t = round(cv_t,4)
print("t-statistic = {}".format(t_stat))
print("t-critical = {}".format(cv_t))


# In[48]:


#visualize data
newName = "Total_Customers"
#data of CONTROL STORE
tmp_Controldf = controlStoredata.reset_index()
tmp_Controldf = tmp_Controldf[['MONTH_ID',columnName]]
tmp_Controldf['Store_Type']="Control_Store"
tmp_Controldf.rename(columns={'MONTH_ID':"Transaction_Month",columnName:newName},inplace=True)
tmp_Controldf.head()


# In[49]:


#data of TRIAL STORE
tmp_Trialdf = trialstoredata.reset_index()
tmp_Trialdf = tmp_Trialdf[['MONTH_ID',columnName]]
tmp_Trialdf['Store_Type']="Trial_Store"
tmp_Trialdf.rename(columns={'MONTH_ID':"Transaction_Month",columnName:newName},inplace=True)
tmp_Trialdf.head()


# In[50]:


#data of CONTROL store with 5% confidence
pastSales_Control05 = tmp_Controldf.copy()
pastSales_Control05[newName] = pastSales_Control05[newName]  - (stdDev*2)
pastSales_Control05['Store_Type'] = "Control 5 % Confidence"
pastSales_Control05.head()


# In[51]:


#data of CONTROL store with 95% confidence
pastSales_Control95 = tmp_Controldf.copy()
pastSales_Control95[newName] = pastSales_Control95[newName]  + (stdDev*2)
pastSales_Control95['Store_Type'] = "Control 95 % Confidence"
pastSales_Control95.head()


# In[52]:


#merge all data into one dataframe
pastSales = pd.DataFrame(columns=['Store_Type',newName,'Transaction_Month'])
pastSales = pd.concat([pastSales,pastSales_Control05,tmp_Trialdf,tmp_Controldf,pastSales_Control95])
pastSales.reset_index(inplace = True)
pastSales.drop('index',axis=1,inplace=True)
pastSales.info()


# In[53]:


#view data
trialPeriodSales = pastSales[(pastSales['Transaction_Month']<201905) & (pastSales['Transaction_Month']>201901)]
trialPeriodSales


# In[54]:


#delete temporary dataframes
del pastSales_Control05
del pastSales_Control95
del tmp_Controldf
del tmp_Trialdf


# In[55]:


#view data
fig =plt.figure(figsize=(8,3),dpi=100)
ax = fig.add_axes([0,0,1,1])
sns.barplot(y=newName,x='Transaction_Month',hue='Store_Type',data=trialPeriodSales)
plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))


# Observations: the trial in store 77 is significantly different to its control store in the trial period ; This is because the trial store performance lies outside the 5% to 95% confidence interval of the control store in two of the three trial months

# In[56]:


#Store-86 : Performance Analysis
store86 = calcTotalScore(corrSt_86,magnDist86)
print("Control store for Store 86 is Store {}".format(store86))


# In[57]:


store86 = calcTotalScore(corrSt_86,magnDist86)
print("Control store for Store 86 is Store {}".format(store86))


# In[58]:


#change the value of variables
TrialStoreNo = 86
ControlStoreNo = store86


# In[59]:


control_Sales = fetchVal(ControlStoreNo,'TOT_SALES')
control_Cust = fetchVal(ControlStoreNo ,'TOTAL_CUSTOMERS')
trial_Sales = fetchVal(TrialStoreNo,'TOT_SALES')
trial_Cust = fetchVal(TrialStoreNo,'TOTAL_CUSTOMERS')

#create new dataframe 'compartable'
comparTable = pd.DataFrame(index=['TOT_SALES','TOT_CUSTOMER'],columns=['TRIAL_ST','CONTROL_ST'])
comparTable['TRIAL_ST']['TOT_SALES'] = trial_Sales
comparTable['TRIAL_ST']['TOT_CUSTOMER'] = trial_Cust
comparTable['CONTROL_ST']['TOT_SALES'] = control_Sales
comparTable['CONTROL_ST']['TOT_CUSTOMER'] = control_Cust
comparTable=comparTable.transpose()


# In[60]:


#compare data of trial store and control store
x=comparTable
fig =plt.figure(figsize=(9,2.5),dpi=100)
ax = fig.add_axes([0,0,1,1])
plt.subplot(1,2,1)
plt.title("Total Sales comparison")
sns.barplot(x=x.index,y='TOT_SALES',data=x)
plt.subplot(1,2,2)
plt.title("Total Customers comparison")
sns.barplot(x=x.index,y='TOT_CUSTOMER',data=x)
plt.tight_layout()


# In[61]:


#Sales of Store-86
columnName = 'TOT_SALES'


# In[62]:


#scaling
scalingFactor = calcScalingFactor(TrialStoreNo,ControlStoreNo,columnName)
scalingFactor


# In[63]:


#copies
controlStoredata = allTimeData.loc[ControlStoreNo].copy()
trialstoredata = allTimeData.loc[TrialStoreNo].copy()


# In[64]:


#after scaling
scaleddata=controlStoredata[[columnName]]*scalingFactor
scaleddata.head()


# In[65]:


#percent diff
percentDiff_Sales = abs(scaleddata[[columnName]]-trialstoredata[[columnName]])*100/scaleddata[[columnName]]
percentDiff_Sales.head()


# In[66]:


#mean/std. deviation
meanControl = scaleddata.loc[201902:201904].copy()
m_Cont = meanControl[columnName].mean()
m_Cont = round(m_Cont,2)

meanTrial = trialstoredata.loc[201902:201904].copy()
m_Trial = meanTrial[columnName].mean()
m_Trial = round(m_Trial,2)

stdDev = (percentDiff_Sales.loc[:201901]).std()
stdDev = stdDev[columnName]
stdDev = round(stdDev,4)

print("Mean of scaled control store = {}".format(m_Cont))
print("Mean of Trial store = {}".format(m_Trial))
print("Standard deviation of data = {}".format(stdDev))


# In[67]:


#perform t-test for alpha=0.05
from scipy.stats import t
#Calculating t-statistic
t_stat = abs(m_Cont-m_Trial)/stdDev
t_stat = round(t_stat,4)
#calculate critical t-value
alpha=0.05
degree = 8-1
cv_t = t.ppf(1-alpha, degree)
cv_t = round(cv_t,4)
print("t-statistic = {}".format(t_stat))
print("t-critical = {}".format(cv_t))


# In[68]:


newName = "Total_Sales"
#data of CONTROL STORE
tmp_Controldf = controlStoredata.reset_index()
tmp_Controldf = tmp_Controldf[['MONTH_ID',columnName]]
tmp_Controldf['Store_Type']="Control_Store"
tmp_Controldf.rename(columns={'MONTH_ID':"Transaction_Month",columnName:newName},inplace=True)
tmp_Controldf.head()


# In[69]:


#data of TRIAL STORE
tmp_Trialdf = trialstoredata.reset_index()
tmp_Trialdf = tmp_Trialdf[['MONTH_ID',columnName]]
tmp_Trialdf['Store_Type']="Trial_Store"
tmp_Trialdf.rename(columns={'MONTH_ID':"Transaction_Month",columnName:newName},inplace=True)
tmp_Trialdf.head()


# In[70]:


#data of CONTROL store with 95% confidence
pastSales_Control05 = tmp_Controldf.copy()
pastSales_Control05[newName] = pastSales_Control05[newName]  - (stdDev*2)
pastSales_Control05['Store_Type'] = "Control 5 % Confidence"
pastSales_Control05.head()


# In[71]:


#data of CONTROL store with 5% confidence
pastSales_Control95 = tmp_Controldf.copy()
pastSales_Control95[newName] = pastSales_Control95[newName]  + (stdDev*2)
pastSales_Control95['Store_Type'] = "Control 95 % Confidence"
pastSales_Control95.head()


# In[72]:


#merge all data into one dataframe
pastSales = pd.DataFrame(columns=['Store_Type',newName,'Transaction_Month'])
pastSales = pd.concat([pastSales,pastSales_Control05,tmp_Trialdf,tmp_Controldf,pastSales_Control95])
pastSales.reset_index(inplace = True)
pastSales.drop('index',axis=1,inplace=True)
pastSales.info()


# In[73]:


#view data
trialPeriodSales = pastSales[(pastSales['Transaction_Month']<201905) & (pastSales['Transaction_Month']>201901)]
trialPeriodSales


# In[74]:


#delete temporary dataframes
del pastSales_Control05
del pastSales_Control95
del tmp_Controldf
del tmp_Trialdf


# In[75]:


#view data
fig =plt.figure(figsize=(8,3),dpi=100)
ax = fig.add_axes([0,0,1,1])
sns.barplot(y=newName,x='Transaction_Month',hue='Store_Type',data=trialPeriodSales)
plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))


# Observations: the trial in store 86 is not significantly different to its control store in the trial period as the trial store performance almost lies inside the 5% to 95% confidence interval of the control store in two of the three trial months

# In[76]:


#Customers of Store-86
columnName = 'TOTAL_CUSTOMERS'


# In[77]:


#scaling
scalingFactor = calcScalingFactor(TrialStoreNo,ControlStoreNo,columnName)
scalingFactor


# In[78]:


#make data frames to be used later
controlStoredata = allTimeData.loc[ControlStoreNo].copy()
trialstoredata = allTimeData.loc[TrialStoreNo].copy()


# In[79]:


#after scaling
scaleddata=controlStoredata[[columnName]]*scalingFactor
scaleddata.head()


# In[80]:


#percent diff
percentDiff_Sales = abs(scaleddata[[columnName]]-trialstoredata[[columnName]])*100/scaleddata[[columnName]]
percentDiff_Sales.head()


# In[81]:


#mean/std. deviation
meanControl = scaleddata.loc[201902:201904].copy()
m_Cont = meanControl[columnName].mean()
m_Cont = round(m_Cont,2)

meanTrial = trialstoredata.loc[201902:201904].copy()
m_Trial = meanTrial[columnName].mean()
m_Trial = round(m_Trial,2)

stdDev = (percentDiff_Sales.loc[:201901]).std()
stdDev = stdDev[columnName]
stdDev = round(stdDev,4)

print("Mean of scaled control store = {}".format(m_Cont))
print("Mean of Trial store = {}".format(m_Trial))
print("Standard deviation of data = {}".format(stdDev))


# In[82]:


#perform t-test for alpha=0.05
from scipy.stats import t
#calculate t-statistic
t_stat = abs(m_Cont-m_Trial)/stdDev
t_stat = round(t_stat,4)
#calculate critical t-value
alpha=0.05
degree = 8-1
cv_t = t.ppf(1-alpha, degree)
cv_t = round(cv_t,4)
print("t-statistic = {}".format(t_stat))
print("t-critical = {}".format(cv_t))


# In[83]:


#visualize data
newName = "Total_Customers"
#data of CONTROL STORE
tmp_Controldf = controlStoredata.reset_index()
tmp_Controldf = tmp_Controldf[['MONTH_ID',columnName]]
tmp_Controldf['Store_Type']="Control_Store"
tmp_Controldf.rename(columns={'MONTH_ID':"Transaction_Month",columnName:newName},inplace=True)
tmp_Controldf.head()


# In[84]:


#data of TRIAL STORE
tmp_Trialdf = trialstoredata.reset_index()
tmp_Trialdf = tmp_Trialdf[['MONTH_ID',columnName]]
tmp_Trialdf['Store_Type']="Trial_Store"
tmp_Trialdf.rename(columns={'MONTH_ID':"Transaction_Month",columnName:newName},inplace=True)
tmp_Trialdf.head()


# In[85]:


#data of CONTROL store with 5% confidence
pastSales_Control05 = tmp_Controldf.copy()
pastSales_Control05[newName] = pastSales_Control05[newName]  - (stdDev*2)
pastSales_Control05['Store_Type'] = "Control 5 % Confidence"
pastSales_Control05.head()


# In[86]:


#data of CONTROL store with 95% confidence
pastSales_Control95 = tmp_Controldf.copy()
pastSales_Control95[newName] = pastSales_Control95[newName]  + (stdDev*2)
pastSales_Control95['Store_Type'] = "Control 95 % Confidence"
pastSales_Control95.head()


# In[87]:


#merge all data into one dataframe
pastSales = pd.DataFrame(columns=['Store_Type',newName,'Transaction_Month'])
pastSales = pd.concat([pastSales,pastSales_Control05,tmp_Trialdf,tmp_Controldf,pastSales_Control95])
pastSales.reset_index(inplace = True)
pastSales.drop('index',axis=1,inplace=True)
pastSales.info()


# In[88]:


#view data
trialPeriodSales = pastSales[(pastSales['Transaction_Month']<201905) & (pastSales['Transaction_Month']>201901)]
trialPeriodSales


# In[89]:


#deleting temporary dataframes
del pastSales_Control05
del pastSales_Control95
del tmp_Controldf
del tmp_Trialdf


# In[90]:


#view data
fig =plt.figure(figsize=(8,3),dpi=100)
ax = fig.add_axes([0,0,1,1])
sns.barplot(y=newName,x='Transaction_Month',hue='Store_Type',data=trialPeriodSales)
plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))


# Observations: The number of customers is significantly higher in all of the three months. This seems to suggest that the trial had a significant impact on increasing the number of customers in trial store 86 but as we saw, sales were not significantly higher. We should check with the Category Manager if there were special deals in the trial store that were may have resulted in lower prices, impacting the results.

# In[91]:


#Store-88 : Performance Analysis
#calculate total score and find control center
def calcScore(corrTbl,magnTbl):
    combinedScoreTable = 0
    #After merging all 4 scores
    #combinedScoreTable = pd.concat([corrTbl,magnTbl],axis=1)
    combinedScoreTable = pd.concat([magnTbl],axis=1)
    #Calculating total score
    combinedScoreTable['Total_Score']=combinedScoreTable.mean(axis=1)
    #Maximum score
    storeNBR = combinedScoreTable[combinedScoreTable['Total_Score']==combinedScoreTable['Total_Score'].max()].index[0]
    return storeNBR


# In[92]:


store88 = calcScore(corrSt_88,magnDist88)
print("Control store for Store 88 is Store {}".format(store88))


# In[93]:


#change the value of variables
TrialStoreNo = 88
ControlStoreNo = store88


# In[94]:


control_Sales = fetchVal(ControlStoreNo,'TOT_SALES')
control_Cust = fetchVal(ControlStoreNo ,'TOTAL_CUSTOMERS')
trial_Sales = fetchVal(TrialStoreNo,'TOT_SALES')
trial_Cust = fetchVal(TrialStoreNo,'TOTAL_CUSTOMERS')

#create new dataframe 'compartable'
comparTable = pd.DataFrame(index=['TOT_SALES','TOT_CUSTOMER'],columns=['TRIAL_ST','CONTROL_ST'])
comparTable['TRIAL_ST']['TOT_SALES'] = trial_Sales
comparTable['TRIAL_ST']['TOT_CUSTOMER'] = trial_Cust
comparTable['CONTROL_ST']['TOT_SALES'] = control_Sales
comparTable['CONTROL_ST']['TOT_CUSTOMER'] = control_Cust
comparTable=comparTable.transpose()


# In[95]:


#Comparing data of trial store and control store
x=comparTable
fig =plt.figure(figsize=(9,2.5),dpi=100)
ax = fig.add_axes([0,0,1,1])
plt.subplot(1,2,1)
plt.title("Total Sales comparison")
sns.barplot(x=x.index,y='TOT_SALES',data=x)
plt.subplot(1,2,2)
plt.title("Total Customers comparison")
sns.barplot(x=x.index,y='TOT_CUSTOMER',data=x)
plt.tight_layout()


# In[96]:


#Sales of Store-88
columnName = 'TOT_SALES'


# In[97]:


#scaling
scalingFactor = calcScalingFactor(TrialStoreNo,ControlStoreNo,columnName)
scalingFactor


# In[98]:


#copies
controlStoredata = allTimeData.loc[ControlStoreNo].copy()
trialstoredata = allTimeData.loc[TrialStoreNo].copy()


# In[99]:


#after scaling
scaleddata=controlStoredata[[columnName]]*scalingFactor
scaleddata.head()


# In[100]:


#percent diff
percentDiff_Sales = abs(scaleddata[[columnName]]-trialstoredata[[columnName]])*100/scaleddata[[columnName]]
percentDiff_Sales.head()


# In[101]:


#mean/std. deviation
meanControl = scaleddata.loc[201902:201904].copy()
m_Cont = meanControl[columnName].mean()
m_Cont = round(m_Cont,2)

meanTrial = trialstoredata.loc[201902:201904].copy()
m_Trial = meanTrial[columnName].mean()
m_Trial = round(m_Trial,2)

stdDev = (percentDiff_Sales.loc[:201901]).std()
stdDev = stdDev[columnName]
stdDev = round(stdDev,4)

print("Mean of scaled control store = {}".format(m_Cont))
print("Mean of Trial store = {}".format(m_Trial))
print("Standard deviation of data = {}".format(stdDev))


# In[102]:


#perform t-test for alpha=0.05
from scipy.stats import t
#Calculating t-statistic
t_stat = abs(m_Cont-m_Trial)/stdDev
t_stat = round(t_stat,4)
#calculate critical t-value
alpha=0.05
degree = 8-1
cv_t = t.ppf(1-alpha, degree)
cv_t = round(cv_t,4)
print("t-statistic = {}".format(t_stat))
print("t-critical = {}".format(cv_t))


# In[103]:


newName = "Total_Sales"
#data of CONTROL STORE
tmp_Controldf = controlStoredata.reset_index()
tmp_Controldf = tmp_Controldf[['MONTH_ID',columnName]]
tmp_Controldf['Store_Type']="Control_Store"
tmp_Controldf.rename(columns={'MONTH_ID':"Transaction_Month",columnName:newName},inplace=True)
tmp_Controldf.head()


# In[104]:


#data of TRIAL STORE
tmp_Trialdf = trialstoredata.reset_index()
tmp_Trialdf = tmp_Trialdf[['MONTH_ID',columnName]]
tmp_Trialdf['Store_Type']="Trial_Store"
tmp_Trialdf.rename(columns={'MONTH_ID':"Transaction_Month",columnName:newName},inplace=True)
tmp_Trialdf.head()


# In[105]:


#data of CONTROL store with 5% confidence
pastSales_Control05 = tmp_Controldf.copy()
pastSales_Control05[newName] = pastSales_Control05[newName]  - (stdDev*2)
pastSales_Control05['Store_Type'] = "Control 5 % Confidence"
pastSales_Control05.head()


# In[106]:


#data of CONTROL store with 95% confidence
pastSales_Control95 = tmp_Controldf.copy()
pastSales_Control95[newName] = pastSales_Control95[newName]  + (stdDev*2)
pastSales_Control95['Store_Type'] = "Control 95 % Confidence"
pastSales_Control95.head()


# In[107]:


#merge all data into one dataframe
pastSales = pd.DataFrame(columns=['Store_Type',newName,'Transaction_Month'])
pastSales = pd.concat([pastSales,pastSales_Control05,tmp_Trialdf,tmp_Controldf,pastSales_Control95])
pastSales.reset_index(inplace = True)
pastSales.drop('index',axis=1,inplace=True)
pastSales.info()


# In[108]:


#view data
trialPeriodSales = pastSales[(pastSales['Transaction_Month']<201905) & (pastSales['Transaction_Month']>201901)]
trialPeriodSales


# In[109]:


#Deleting temporary dataframes
del pastSales_Control05
del pastSales_Control95
del tmp_Controldf
del tmp_Trialdf


# In[110]:


#view data
fig =plt.figure(figsize=(8,3),dpi=100)
ax = fig.add_axes([0,0,1,1])
sns.barplot(y=newName,x='Transaction_Month',hue='Store_Type',data=trialPeriodSales)
plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))


# Observations: the trial in store 88 is significantly different to its control store in the trial period as the trial store performance lies outside of the 5% to 95% confidence interval of the control store in two of the three trial months.

# In[111]:


#Customers of Store-88
columnName = 'TOTAL_CUSTOMERS'


# In[112]:


#scaling
scalingFactor = calcScalingFactor(TrialStoreNo,ControlStoreNo,columnName)
scalingFactor


# In[113]:


#data frames to be used later
controlStoredata = allTimeData.loc[ControlStoreNo].copy()
trialstoredata = allTimeData.loc[TrialStoreNo].copy()


# In[114]:


#after scaling
scaleddata=controlStoredata[[columnName]]*scalingFactor
scaleddata.head()


# In[115]:


#percent diff
percentDiff_Sales = abs(scaleddata[[columnName]]-trialstoredata[[columnName]])*100/scaleddata[[columnName]]
percentDiff_Sales.head()


# In[116]:


#Finding means and std. deviation
meanControl = scaleddata.loc[201902:201904].copy()
m_Cont = meanControl[columnName].mean()
m_Cont = round(m_Cont,2)

meanTrial = trialstoredata.loc[201902:201904].copy()
m_Trial = meanTrial[columnName].mean()
m_Trial = round(m_Trial,2)

stdDev = (percentDiff_Sales.loc[:201901]).std()
stdDev = stdDev[columnName]
stdDev = round(stdDev,4)

print("Mean of scaled control store = {}".format(m_Cont))
print("Mean of Trial store = {}".format(m_Trial))
print("Standard deviation of data = {}".format(stdDev))


# In[117]:


# Performing t-test for alpha=0.05
from scipy.stats import t
#Calculating t-statistic
t_stat = abs(m_Cont-m_Trial)/stdDev
t_stat = round(t_stat,4)
#Calculating critical t-value
alpha=0.05
degree = 8-1
cv_t = t.ppf(1-alpha, degree)
cv_t = round(cv_t,4)
print("t-statistic = {}".format(t_stat))
print("t-critical = {}".format(cv_t))


# In[118]:


# To visualize data
newName = "Total_Customers"
# Data of CONTROL STORE
tmp_Controldf = controlStoredata.reset_index()
tmp_Controldf = tmp_Controldf[['MONTH_ID',columnName]]
tmp_Controldf['Store_Type']="Control_Store"
tmp_Controldf.rename(columns={'MONTH_ID':"Transaction_Month",columnName:newName},inplace=True)
tmp_Controldf.head()


# In[119]:


# Data of TRIAL STORE
tmp_Trialdf = trialstoredata.reset_index()
tmp_Trialdf = tmp_Trialdf[['MONTH_ID',columnName]]
tmp_Trialdf['Store_Type']="Trial_Store"
tmp_Trialdf.rename(columns={'MONTH_ID':"Transaction_Month",columnName:newName},inplace=True)
tmp_Trialdf.head()


# In[120]:


# Data of CONTROL store with 5% confidence
pastSales_Control05 = tmp_Controldf.copy()
pastSales_Control05[newName] = pastSales_Control05[newName]  - (stdDev*2)
pastSales_Control05['Store_Type'] = "Control 5 % Confidence"
pastSales_Control05.head()


# In[121]:


# Data of CONTROL store with 95% confidence
pastSales_Control95 = tmp_Controldf.copy()
pastSales_Control95[newName] = pastSales_Control95[newName]  + (stdDev*2)
pastSales_Control95['Store_Type'] = "Control 95 % Confidence"
pastSales_Control95.head()


# In[122]:


# Merging all data into one dataframe
pastSales = pd.DataFrame(columns=['Store_Type',newName,'Transaction_Month'])
pastSales = pd.concat([pastSales,pastSales_Control05,tmp_Trialdf,tmp_Controldf,pastSales_Control95])
pastSales.reset_index(inplace = True)
pastSales.drop('index',axis=1,inplace=True)
pastSales.info()


# In[123]:


trialPeriodSales = pastSales[(pastSales['Transaction_Month']<201905) & (pastSales['Transaction_Month']>201901)]
trialPeriodSales


# In[124]:


#Deleting temporary dataframes
del pastSales_Control05
del pastSales_Control95
del tmp_Controldf
del tmp_Trialdf


# In[125]:


fig =plt.figure(figsize=(8,3),dpi=100)
ax = fig.add_axes([0,0,1,1])
sns.barplot(y=newName,x='Transaction_Month',hue='Store_Type',data=trialPeriodSales)
plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))


# Observations: Total number of customers in the trial period for the trial store is significantly higher than the control store for two out of three months, which indicates a positive trial effect.

# Conclusion:
# 
# Control stores 233, 155, 237 for trial stores 77, 86 and 88 respectively.
# The results for trial stores 77 and 88 during the trial period show a significant difference in at least two of the three trial months but this is not the case for trial store 86.

# In[ ]:




