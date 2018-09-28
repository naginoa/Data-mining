#coding=utf-8
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy import interpolate
import seaborn as sns
from scipy import interpolate
import time
from utils import *

train_path='./data/round1_iflyad_train.txt'
test_path='./data/round1_iflyad_test_feature.txt'


all_data=pd.read_table(train_path)
#print(all_data.head(10))
all_test=pd.read_table(test_path)
#print(all_test.head(10))
#将时间戳转化为正常时间
all_data['time_string']=all_data["time"].apply(lambda x:time.strftime("%Y-%m-%d %H:%M:%S",time.localtime(x)))
all_data['time_string']=pd.to_datetime(all_data["time_string"])
all_data["hour"]=all_data["time_string"].dt.hour
all_data["day"]=all_data["time_string"].dt.day
all_data["day"]=all_data["day"].apply(lambda x:x-27 if x>=27 else x+4)

all_test['time_string']=all_test["time"].apply(lambda x:time.strftime("%Y-%m-%d %H:%M:%S",time.localtime(x)))
all_test['time_string']=pd.to_datetime(all_test["time_string"])
all_test["hour"]=all_test["time_string"].dt.hour
all_test["day"]=all_test["time_string"].dt.day
all_test["day"]=all_test["day"].apply(lambda x:x-27 if x>=27 else x+4)

#划分训练集与测试集
#27，28，29，30，31，1，2，3
'''
          	          feature_extract_dataset(day)    label     
train1                0-4                             5
train2(offline_test)  1-5                             6
online_test           2-6                             7
'''
features1=all_data[(all_data['day']>=0)&(all_data['day']<=4)]
dataset1=all_data[all_data['day']==5]
print(dataset1.shape)
features2=all_data[(all_data['day']>=1)&(all_data['day']<=5)]
dataset2=all_data[all_data['day']==6]
features3=all_data[(all_data['day']>=2)&(all_data['day']<=6)]
dataset3=all_test
'''
aa=all_data[['inner_slot_id']]
bb=all_test[['inner_slot_id']]
cc=pd.concat([aa,bb],axis=0)
cc.drop_duplicates(inplace=True)
df=pd.get_dummies(cc['inner_slot_id'],prefix='inner_slot_id')
inner_slot_id_one_hot=pd.concat([cc,df],axis=1)
'''
#构造特征
#点击个数，转化率，转化个数
t1=features1[['adid','click']]
t1=get_type_features(t1,['adid'],'click',"sum",'adid_click_num')
t1=get_type_features(t1,['adid'],'click',"count",'adid_click_cnt')
t1=get_type_features(t1,['adid'],'click',"mean",'adid_click_radio')
t11=t1[['adid','adid_click_num','adid_click_cnt','adid_click_radio']]
t11.drop_duplicates(inplace=True)
print(t1.head(10))
t2=features1[['app_id','click']]
t2=get_type_features(t2,['app_id'],'click',"sum",'appid_click_num')
t2=get_type_features(t2,['app_id'],'click',"count",'appid_click_cnt')
t2=get_type_features(t2,['app_id'],'click',"mean",'appid_click_radio')
t21=t2[['app_id','appid_click_num','appid_click_cnt','appid_click_radio']]
t21.drop_duplicates(inplace=True)

t3=features1[['app_id','adid','click']]
t3=get_type_features(t3,['app_id','adid'],'click',"sum",'appid_ad_click_num')
t3=get_type_features(t3,['app_id','adid'],'click',"count",'appid_ad_click_cnt')
t3=get_type_features(t3,['app_id','adid'],'click',"mean",'appid_ad_click_radio')
t31=t3[['app_id','adid','appid_ad_click_num','appid_ad_click_cnt','appid_ad_click_radio']]
t31.drop_duplicates(inplace=True)

t4=features1[['orderid','click']]
t4=get_type_features(t4,['orderid'],'click',"sum",'orderid_click_num')
t4=get_type_features(t4,['orderid'],'click',"count",'orderid_click_cnt')
t4=get_type_features(t4,['orderid'],'click',"mean",'orderid_click_radio')
t41=t4[['orderid','orderid_click_num','orderid_click_cnt','orderid_click_radio']]
t41.drop_duplicates(inplace=True)

t5=features1[['inner_slot_id','click']]
t5=get_type_features(t5,['inner_slot_id'],'click',"sum",'inner_slot_id_click_num')
t5=get_type_features(t5,['inner_slot_id'],'click',"count",'inner_slot_id_click_cnt')
t5=get_type_features(t5,['inner_slot_id'],'click',"mean",'inner_slot_id_click_radio')
t51=t5[['inner_slot_id','inner_slot_id_click_num','inner_slot_id_click_cnt','inner_slot_id_click_radio']]
t51.drop_duplicates(inplace=True)

t6=features1[['inner_slot_id','nnt','click']]
t6=get_type_features(t6,['inner_slot_id','nnt'],'click',"sum",'inner_slot_id_nnt_click_num')
t6=get_type_features(t6,['inner_slot_id','nnt'],'click',"count",'inner_slot_id_nnt_click_cnt')
t6=get_type_features(t6,['inner_slot_id','nnt'],'click',"mean",'inner_slot_id_nnt_click_radio')
t61=t6[['inner_slot_id','nnt','inner_slot_id_nnt_click_num','inner_slot_id_nnt_click_cnt','inner_slot_id_nnt_click_radio']]
t61.drop_duplicates(inplace=True)


dataset1=dataset1.merge(t11,on=['adid'],how='left')
dataset1=dataset1.merge(t21,on=['app_id'],how='left')
dataset1=dataset1.merge(t31,on=['app_id','adid'],how='left')
dataset1=dataset1.merge(t41,on=['orderid'],how='left')
dataset1=dataset1.merge(t51,on=['inner_slot_id'],how='left')
dataset1=dataset1.merge(t61,on=['inner_slot_id','nnt'],how='left')
#dataset1=dataset1.merge(inner_slot_id_one_hot,on=['inner_slot_id'],how='left')
#dataset1['creative_shape']=dataset1['creative_height']*dataset1['creative_width']
#保存提取的特征
dataset1.to_csv('features/feature1.csv',index=None)
#############################################################################################
#features2
t1=features2[['adid','click']]
t1=get_type_features(t1,['adid'],'click',"sum",'adid_click_num')
t1=get_type_features(t1,['adid'],'click',"count",'adid_click_cnt')
t1=get_type_features(t1,['adid'],'click',"mean",'adid_click_radio')
t11=t1[['adid','adid_click_num','adid_click_cnt','adid_click_radio']]
t11.drop_duplicates(inplace=True)
#print(t1.head(10))
t2=features2[['app_id','click']]
t2=get_type_features(t2,['app_id'],'click',"sum",'appid_click_num')
t2=get_type_features(t2,['app_id'],'click',"count",'appid_click_cnt')
t2=get_type_features(t2,['app_id'],'click',"mean",'appid_click_radio')
t21=t2[['app_id','appid_click_num','appid_click_cnt','appid_click_radio']]
t21.drop_duplicates(inplace=True)

t3=features2[['app_id','adid','click']]
t3=get_type_features(t3,['app_id','adid'],'click',"sum",'appid_ad_click_num')
t3=get_type_features(t3,['app_id','adid'],'click',"count",'appid_ad_click_cnt')
t3=get_type_features(t3,['app_id','adid'],'click',"mean",'appid_ad_click_radio')
t31=t3[['app_id','adid','appid_ad_click_num','appid_ad_click_cnt','appid_ad_click_radio']]
t31.drop_duplicates(inplace=True)

t4=features2[['orderid','click']]
t4=get_type_features(t4,['orderid'],'click',"sum",'orderid_click_num')
t4=get_type_features(t4,['orderid'],'click',"count",'orderid_click_cnt')
t4=get_type_features(t4,['orderid'],'click',"mean",'orderid_click_radio')
t41=t4[['orderid','orderid_click_num','orderid_click_cnt','orderid_click_radio']]
t41.drop_duplicates(inplace=True)

t5=features2[['inner_slot_id','click']]
t5=get_type_features(t5,['inner_slot_id'],'click',"sum",'inner_slot_id_click_num')
t5=get_type_features(t5,['inner_slot_id'],'click',"count",'inner_slot_id_click_cnt')
t5=get_type_features(t5,['inner_slot_id'],'click',"mean",'inner_slot_id_click_radio')
t51=t5[['inner_slot_id','inner_slot_id_click_num','inner_slot_id_click_cnt','inner_slot_id_click_radio']]
t51.drop_duplicates(inplace=True)

t6=features2[['inner_slot_id','nnt','click']]
t6=get_type_features(t6,['inner_slot_id','nnt'],'click',"sum",'inner_slot_id_nnt_click_num')
t6=get_type_features(t6,['inner_slot_id','nnt'],'click',"count",'inner_slot_id_nnt_click_cnt')
t6=get_type_features(t6,['inner_slot_id','nnt'],'click',"mean",'inner_slot_id_nnt_click_radio')
t61=t6[['inner_slot_id','nnt','inner_slot_id_nnt_click_num','inner_slot_id_nnt_click_cnt','inner_slot_id_nnt_click_radio']]
t61.drop_duplicates(inplace=True)
#one-hot
#inner_slot_id_df=pd.get_dummies(dataset2['inner_slot_id'],prefix='inner_slot_id')
dataset2=dataset2.merge(t11,on=['adid'],how='left')
dataset2=dataset2.merge(t21,on=['app_id'],how='left')
dataset2=dataset2.merge(t31,on=['app_id','adid'],how='left')
dataset2=dataset2.merge(t41,on=['orderid'],how='left')
dataset2=dataset2.merge(t51,on=['inner_slot_id'],how='left')
dataset2=dataset2.merge(t61,on=['inner_slot_id','nnt'],how='left')
#dataset2=dataset2.merge(inner_slot_id_one_hot,on=['inner_slot_id'],how='left')
#dataset2['creative_shape']=dataset2['creative_height']*dataset2['creative_width']

dataset2.to_csv('features/feature2.csv',index=None)
##################################################################################################
#test数据集
#features3
t1=features3[['adid','click']]
t1=get_type_features(t1,['adid'],'click',"sum",'adid_click_num')
t1=get_type_features(t1,['adid'],'click',"count",'adid_click_cnt')
t1=get_type_features(t1,['adid'],'click',"mean",'adid_click_radio')
t11=t1[['adid','adid_click_num','adid_click_cnt','adid_click_radio']]
t11.drop_duplicates(inplace=True)
print(t1.head(10))
t2=features3[['app_id','click']]
t2=get_type_features(t2,['app_id'],'click',"sum",'appid_click_num')
t2=get_type_features(t2,['app_id'],'click',"count",'appid_click_cnt')
t2=get_type_features(t2,['app_id'],'click',"mean",'appid_click_radio')
t21=t2[['app_id','appid_click_num','appid_click_cnt','appid_click_radio']]
t21.drop_duplicates(inplace=True)

t3=features3[['app_id','adid','click']]
t3=get_type_features(t3,['app_id','adid'],'click',"sum",'appid_ad_click_num')
t3=get_type_features(t3,['app_id','adid'],'click',"count",'appid_ad_click_cnt')
t3=get_type_features(t3,['app_id','adid'],'click',"mean",'appid_ad_click_radio')
t31=t3[['app_id','adid','appid_ad_click_num','appid_ad_click_cnt','appid_ad_click_radio']]
t31.drop_duplicates(inplace=True)

t4=features3[['orderid','click']]
t4=get_type_features(t4,['orderid'],'click',"sum",'orderid_click_num')
t4=get_type_features(t4,['orderid'],'click',"count",'orderid_click_cnt')
t4=get_type_features(t4,['orderid'],'click',"mean",'orderid_click_radio')
t41=t4[['orderid','orderid_click_num','orderid_click_cnt','orderid_click_radio']]
t41.drop_duplicates(inplace=True)

t5=features3[['inner_slot_id','click']]
t5=get_type_features(t5,['inner_slot_id'],'click',"sum",'inner_slot_id_click_num')
t5=get_type_features(t5,['inner_slot_id'],'click',"count",'inner_slot_id_click_cnt')
t5=get_type_features(t5,['inner_slot_id'],'click',"mean",'inner_slot_id_click_radio')
t51=t5[['inner_slot_id','inner_slot_id_click_num','inner_slot_id_click_cnt','inner_slot_id_click_radio']]
t51.drop_duplicates(inplace=True)

t6=features3[['inner_slot_id','nnt','click']]
t6=get_type_features(t6,['inner_slot_id','nnt'],'click',"sum",'inner_slot_id_nnt_click_num')
t6=get_type_features(t6,['inner_slot_id','nnt'],'click',"count",'inner_slot_id_nnt_click_cnt')
t6=get_type_features(t6,['inner_slot_id','nnt'],'click',"mean",'inner_slot_id_nnt_click_radio')
t61=t6[['inner_slot_id','nnt','inner_slot_id_nnt_click_num','inner_slot_id_nnt_click_cnt','inner_slot_id_nnt_click_radio']]
t61.drop_duplicates(inplace=True)

dataset3=dataset3.merge(t11,on=['adid'],how='left')
dataset3=dataset3.merge(t21,on=['app_id'],how='left')
dataset3=dataset3.merge(t31,on=['app_id','adid'],how='left')
dataset3=dataset3.merge(t41,on=['orderid'],how='left')
dataset3=dataset3.merge(t51,on=['inner_slot_id'],how='left')
dataset3=dataset3.merge(t61,on=['inner_slot_id','nnt'],how='left')
#dataset3=dataset3.merge(inner_slot_id_one_hot,on=['inner_slot_id'],how='left')
#dataset3['creative_shape']=dataset3['creative_height']*dataset3['creative_width']

dataset3.to_csv('features/online_test_features.csv',index=None)