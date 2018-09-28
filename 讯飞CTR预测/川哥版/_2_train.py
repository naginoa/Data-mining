#coding=utf-8
import xgboost as xgb 
import pandas as pd 
import numpy as np 
from sklearn.preprocessing import MinMaxScaler
import datetime

#训练
dataset1 = pd.read_csv('features/feature1.csv')
#dataset1.click.replace(-1,0,inplace=True)
dataset2 = pd.read_csv('features/feature2.csv')
#dataset2.click.replace(-1,0,inplace=True)
dataset3 = pd.read_csv('features/online_test_features.csv')

dataset1.drop_duplicates(inplace=True)
dataset2.drop_duplicates(inplace=True)
dataset3.drop_duplicates(inplace=True)

dataset1= dataset1.replace(np.nan,0)
dataset2= dataset2.replace(np.nan,0)
dataset3= dataset3.replace(np.nan,0)

dataset12 = pd.concat([dataset1,dataset2],axis=0)

dataset1_y = dataset1.click
dataset1_x = dataset1.drop(['instance_id','click','time','time_string','day','user_tags','make','model','advert_industry_inner','advert_name','f_channel','inner_slot_id','osv','os_name'],axis=1)  # 'day_gap_before','day_gap_after' cause overfitting, 0.77
dataset2_y = dataset2.click
dataset2_x = dataset2.drop(['instance_id','click','time','time_string','day','user_tags','make','model','advert_industry_inner','advert_name','f_channel','inner_slot_id','osv','os_name'],axis=1)
dataset12_y = dataset12.click
dataset12_x = dataset12.drop(['instance_id','click','time','time_string','day','user_tags','make','model','advert_industry_inner','advert_name','f_channel','inner_slot_id','osv','os_name'],axis=1)
dataset3_preds = dataset3[['instance_id']]
dataset3_x = dataset3.drop(['instance_id','time','time_string','day','user_tags','make','model','advert_industry_inner','advert_name','f_channel','inner_slot_id','osv','os_name'],axis=1)

print(dataset1_x.shape,dataset2_x.shape,dataset3_x.shape)

dataset1 = xgb.DMatrix(dataset1_x,label=dataset1_y)
dataset2 = xgb.DMatrix(dataset2_x,label=dataset2_y)
dataset12 = xgb.DMatrix(dataset12_x,label=dataset12_y)
dataset3 = xgb.DMatrix(dataset3_x)

params={'booster':'gbtree',
	    'objective': 'binary:logistic',
	    'eval_metric':'logloss',
	    'gamma':0.1,
	    'min_child_weight':1.1,
	    'max_depth':2,
	    'lambda':10,
	    'subsample':0.7,
	    'colsample_bytree':0.7,
	    'colsample_bylevel':0.7,
	    'eta': 0.2,
	    'tree_method':'exact',
	    'seed':0,
	    'nthread':12
	    }

#train on dataset1, evaluate on dataset2
#watchlist = [(dataset1,'train'),(dataset2,'val')]
#model = xgb.train(params,dataset1,num_boost_round=800,evals=watchlist,early_stopping_rounds=300)

watchlist = [(dataset12,'train')]
model = xgb.train(params,dataset12,num_boost_round=3500,evals=watchlist)

#predict test set
dataset3_preds['predicted_score'] = model.predict(dataset3)
#dataset3_preds.click = MinMaxScaler().fit_transform(dataset3_preds.click.reshape(-1, 1))
#dataset3_preds.sort_values(by=['coupon_id','label'],inplace=True)
dataset3_preds.to_csv("./result/xgb_preds.csv",index=None)
print(dataset3_preds.describe())
    
#save feature score
feature_score = model.get_fscore()
feature_score = sorted(feature_score.items(), key=lambda x:x[1],reverse=True)
fs = []
for (key,value) in feature_score:
    fs.append("{0},{1}\n".format(key,value))
    
with open('xgb_feature_score.csv','w') as f:
    f.writelines("feature,score\n")
    f.writelines(fs)