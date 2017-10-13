
# coding: utf-8

# # 数据检查

# In[5]:


import pandas as pd
import numpy as np


# In[18]:


df = pd.read_csv('IrisFishData.csv')
df.head()


# In[19]:


df.describe()


# In[20]:


df.isnull().values.any()


# In[45]:


df = pd.read_csv('IrisFishData.csv', na_values=['NA'])


# df.isnull().sum()

# In[9]:


get_ipython().magic('matplotlib inline')

import matplotlib.pyplot as plt


# In[2]:


import seaborn as sb


# In[24]:


sb.pairplot(df.dropna(), hue = 'class')
#每列的分布在对角线上画出


# # 数据清洗

# In[26]:


df['class'].unique()


# In[46]:


df.loc[df['class'] == 'setossa', 'class'] = 'setosa'
df['class'].unique()


# ## 由图可知蓝色有个点总是在范围外 是不是可能有数据错误

# In[55]:


df.loc[df['class'] == 'setosa', 'sepal_width_cm'].describe()


# In[76]:


df = df.loc[(df['class'] != 'setosa') | ((df['class'] == 'setosa') & (df['sepal_width_cm'] >= 2.5))]
#不是setosa的class的 不需要清洗      是setosa的数据需要过滤到sepal_width_cm 值为2.5以上的 


# In[75]:


sub_df = df.loc[(df['class'] != 'setosa') | (df['sepal_width_cm'] >= 2.5)]
sub_df.loc[sub_df['class'] == 'setosa', 'sepal_width_cm'].hist()


# In[77]:


df.to_csv('Iris_clean.csv', index=False)


# In[6]:


clean_df = pd.read_csv('Iris_clean.csv')


# In[10]:


sb.pairplot(clean_df, hue='class')


# 

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




