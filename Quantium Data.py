#!/usr/bin/env python
# coding: utf-8

# Quantium
# Data preparation and customer analytics

# In[1]:


#imports necessary for analysis
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
pd.set_option('display.max_columns', 1000)
pd.set_option('display.width', 1000)

from pandas import Series
from pandas import DataFrame

import os # accessing directory structure

import numpy as np # linear algebra
from numpy import ndarray
from numpy.random import randn
from numpy import loadtxt, where

import matplotlib as plt
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
plt.rc("font", size=14)
get_ipython().run_line_magic('matplotlib', 'inline')

from mpl_toolkits.mplot3d import Axes3D

import seaborn as sns

import sklearn
import sklearn.preprocessing
from sklearn.preprocessing import StandardScaler

from scipy import stats

import plotly.express as px


# In[2]:


#confirm package versions
print("pandas version " + pd.__version__)
print("numpy version " + np.__version__)
print("seaborn version " + sns.__version__)


# In[3]:


#ignore future warnings 
import warnings 
warnings.filterwarnings('ignore')


# In[4]:


#read customer csv into pandas dataframe
df_customer = pd.read_csv('QVI_purchase_behaviour.csv') 


# In[5]:


#evaluate top of loaded data
df_customer.head()


# In[6]:


#determine data types and get the shape of the data (# of columns and rows)
df_customer.info()


# In[7]:


#evalute for null values
df_customer.isnull().sum()


# In[8]:


#check for duplicates
df_customer.duplicated().sum()


# In[9]:


df_customer.PREMIUM_CUSTOMER.unique()


# In[10]:


#Convert categorical to binary where "Premium"=1 or yes and "Budget" and "Mainstrain" = 0 or no
df_customer['PREMIUM_CUSTOMER'].replace(['Premium', 'Budget', 'Mainstream'],
                        [1, 0, 0], inplace=True)


# In[11]:


#evaluate top of loaded data
df_customer.head()


# In[12]:


#determine data types
df_customer.info()


# In[13]:


#read transaction csv into pandas dataframe
#file converted to csv from .xlsx
df_transaction = pd.read_csv('QVI_transaction_data.csv') 


# In[14]:


#evaluate top of loaded data
df_transaction.head()


# In[15]:


#determine data types and get shape of data (# of columns and rows)
df_transaction.info()


# In[16]:


#convert "DATE" to datetime
df_transaction['DATE'] = pd.to_datetime(df_transaction['DATE'])
#show the change 
df_transaction.info()


# In[17]:


#change "DATE" column display
df_transaction['DATE'] = pd.to_datetime(df_transaction["DATE"].dt.strftime('%Y-%m'))


# In[18]:


#view "DATE" change
df_transaction.head()


# In[19]:


#evalute for null values
df_transaction.isnull().sum()


# In[20]:


#evaluate for duplicate values
df_transaction.duplicated().sum()


# In[21]:


#remove duplicate-use keep='first' to keep the first occurance
df_transaction= df_transaction.drop_duplicates(subset=None, keep="first", inplace=False)


# In[22]:


#evaluate for duplicate values
df_transaction.duplicated().sum()


# In[23]:


#join two csv files at loyalty card number
df = df_transaction.merge(df_customer,on=["LYLTY_CARD_NBR"])
df.head()


# In[24]:


#determine data types and get shape of data
df.info()


# In[25]:


#check for nulls
df.isnull().sum()


# In[26]:


#evaluate for duplicate values
df_transaction.duplicated().sum()


# In[27]:


#save clean data csv
df.to_csv('Quantium_clean_customer_Transaction.csv')


# Analysis

# In[28]:


import pandas_profiling as pp
from pandas_profiling import ProfileReport
from pandas_profiling.utils.cache import cache_file
profile = ProfileReport(df)
profile


# In[29]:


profile.to_notebook_iframe()


# In[30]:


profile.to_file(output_file="quantium_report.html")


# In[31]:


df.skew().sort_values(ascending=False)


# In[32]:


fig=px.scatter(df,
              x="PROD_QTY",
              y="TOT_SALES",
              title="Store area vs Items Available",
              template="plotly_dark")
fig.show()


# In[36]:


df.columns


# In[37]:


fig=px.scatter(df,
              x="LIFESTAGE",
              y="TOT_SALES",
              title="Store area vs Items Available",
              template="plotly_dark")
fig.show()


# In[33]:


sns.distplot(df['TOT_SALES'])

plt.grid()
plt.show()


# In[34]:


sns.pairplot(df)
plt.show()


# Data Cleaning: 
# 
# QVI_purchase_behaviour.csv has 3 columns with 72637 rows. There are 0 duplicates and 0 missing values in this dataset.  
# 
# LYLTY_CARD_NBR is an int64, LIFESTAGE is an object and PREMIUM_CUSTOMER is an object. 
# 
# PREMIUM_CUSTOMER has been converted to binary where "Premium"=1 or yes and "Budget" and "Mainstrain" = 0 or no. 
# 
# PREMIUM_CUSTOMER is now an int64. 
# 
# QVI_transaction_data.csv has 8 columns with 264836 rows. There are no missing values, there was one duplicate value found, the
# 
# last duplicate value was removed from the dataset leaving no duplicates. The column “DATE” was converted from int64 to 
# 
# datetime64, and the column view was converted to year, month, day.
# 
# The two csv files were joined at loyalty card number. There are now 10 unique variables with 264835 observations in the 
# 
# combined dataset. There are zero missing values. There are zero duplicates. There are 3 variable types 1 datetime, 6 numeric, 
# 
# and 3 categorical. 
# 
# Univariate Analysis: 
# 
# "DATE" = there are 12 unique dates in this dataset beginning on 2018-07-01 and ending on 2019-06-01 meaning we have 11 months
# 
# of data in this dataset. 
# 
# “STORE_NBR” = there are 272 unique values aka data on 272 different store locations. 
# 
# “LYLTY_CARD_NBR” = there are 72,637 distinct records aka customers. Some loyalty card numbers are found multiple times in the
# 
# dataset. Below is their count. 
# 
# Loyalty 
# Card #  Count 
# 172032	18
# 
# 162039	18
# 
# 13138	17
# 
# 116181	17
# 
# 128178	17
# 
# 230078	17
# 
# 94185	16
# 
# 129050	16	
# 
# 113080	16	
# 
# 104117	16	 
# 
# These customers used their loyalty card number the most in the given dataset. 
# 
# “PROD_NBR” = there are 114 distinct values aka products in the dataset. 
# 
# Value Count	
# 102	  3304
# 
# 108	  3296
# 
# 33	  3269
# 
# 112	  3268
# 
# 75	  3265	
# 
# 63	  3257
# 
# 74	  3252
# 
# 104	  3242
# 
# 14	  3233
# 
# 28	  3229
# 
# 
# PROD_NBR 114 shows up the most with a count of 3127 
# 
# PROD_NBR 113 shows up 3170 times 
# 
# PRD_NBR 112 shows up 3268 times
# 
# PRD_NBR 109 shows up 3210 times
# 
# PRD_NBR 108 shows up 3296 times 
# 
# 
# PROD_NAME = 
# 
#         Value			                   Count
# 	
# Kettle Mozzarella Basil & Pesto 175g		3304
# 
# Kettle Tortilla ChpsHny&Jlpno Chili 150g	3296
# 
# Cobs Popd Swt/Chlli &Sr/Cream Chips 110g	3269
# 
# Tyrrells Crisps Ched & Chives 165g		    3268
# 
# Cobs Popd Sea Salt Chips 110g			    3265
# 
# Kettle 135g Swt Pot Sea Salt			    3257
# 
# Tostitos Splash Of Lime 175g			    3252
# 
# Infuzions Thai SweetChili PotatoMix 110g	3242
# 
# Smiths Crnkle Chip Orgnl Big Bag 380g		3233
# 
# Thins Potato Chips Hot & Spicy 175g		    3229	 
# 
# “PROD_QTY” = the most frequently occurring product qty is 2 at 82.1% frequency 
# 
# The second most frequently occurring product qty is 1 at 10.4% frequency. 
# 
# The largest PROD_QTY is 200 with a frequency of <0.1% occurring 2x. 
# 
# PROD_QTY of 4 occurs 450 times in the dataset with a frequency of 0.2%. 
# 
# 
# “LIFESTAGE” =  there are 7 distinct life stages in the dataset.
# 
# OLDER SINGLES/COUPLES	54478 
# RETIREES	            49763 
# OLDER FAMILIES	        48596 
# YOUNG FAMILIES	        43592 
# YOUNG SINGLES/COUPLES	36377 
# Other values (2)	    32029 
# 
# 
# 
# PREMIUM_CUSTOMERS = 195,145 of the rows were not associated with a premium customer 
# 
# While 69,690 were associated with a premium customer. There were 125,455 more premium customers in this dataset than non 
# 
# premium customers. 
# 
# Relationships: 
# 
# PROD_NAME has a high cardinality: 114 distinct values	                High cardinality
# 
# STORE_NBR is highly correlated with LYLTY_CARD_NBR and 1 other fields	High correlation
# 
# LYLTY_CARD_NBR is highly correlated with STORE_NBR and 1 other fields	High correlation
# 
# TXN_ID is highly correlated with STORE_NBR and 1 other fields	        High correlation
# 
# STORE_NBR is highly correlated with LYLTY_CARD_NBR and 1 other fields	High correlation
# 
# LYLTY_CARD_NBR is highly correlated with STORE_NBR and 1 other fields	High correlation
# 
# TXN_ID is highly correlated with STORE_NBR and 1 other fields	        High correlation
# 
# PROD_QTY is highly correlated with TOT_SALES	                        High correlation
# 
# TOT_SALES is highly correlated with PROD_QTY	                        High correlation
# 
# STORE_NBR is highly correlated with LYLTY_CARD_NBR and 1 other fields	High correlation
# 
# LYLTY_CARD_NBR is highly correlated with STORE_NBR and 1 other fields	High correlation
# 
# TXN_ID is highly correlated with STORE_NBR and 1 other fields	        High correlation
# 
# STORE_NBR is highly correlated with LYLTY_CARD_NBR and 1 other fields	High correlation
# 
# LYLTY_CARD_NBR is highly correlated with STORE_NBR and 1 other fields	High correlation
# 
# TXN_ID is highly correlated with STORE_NBR and 1 other fields	        High correlation
# 
# PROD_QTY is highly correlated with TOT_SALES	                        High correlation
# 
# TOT_SALES is highly correlated with PROD_QTY	                        High correlation
# 
# PROD_QTY is highly skewed (γ1 = 220.1026848)	                        Skewed
# 
# TOT_SALES is highly skewed (γ1 = 68.56956685)	                        Skewed
# 
# 
# Customers tend to shop at the same stores often as there is high correlation between store number and loyalty card number. 
# 
# Total sales and product quantity are highly correlated. 
# 

# In[ ]:




