
# coding: utf-8

# # 数据探索

# In[2]:


get_ipython().magic('matplotlib inline')

import matplotlib.pyplot as plt 
import pandas as pd 
import numpy as np 
import seaborn as sb


# In[3]:


df = pd.read_csv('Iris_clean.csv')
sb.pairplot(df)


# In[4]:


plt.figure(figsize=(10,10))

for column_index, column in enumerate(df.columns):
    if column == 'class':
        continue
    #按照索引分成四个小图
    plt.subplot(2, 2, column_index+1)
    #在每个小图上画出特征
    sb.violinplot(x='class', y=column, data=df)


# ### 测试训练集

# In[5]:


df = pd.read_csv('Iris_clean.csv')
#scikit-learn 需要输入的是numpy的array 
training_set = df[['sepal_lenth_cm','sepal_width_cm','petal_length_cm','petal_width_cm']].values
print(training_set[:5])
training_class = df['class'].values
print(training_class[:5])


# In[11]:


from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
#忽略这些warning才可以进行训练


# In[12]:


(training_inputs,
testing_inputs,
training_classes,
testing_classes) = train_test_split(training_set, training_class, train_size=0.75, random_state=1)


# from sklearn.tree import DecisionTreeClassifier
# 
# #创分类器对象
# tree_classfier = DecisionTreeClassifier()
# 
# tree_classfier.fit(training_inputs, training_classes)
# tree_classfier.score(testing_inputs, testing_classes)

# ## 97%的正确率  还不错

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




