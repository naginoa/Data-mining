#coding=utf-8
import pandas as pd 

#获取统计特征，包括点击次数，转化次数，转化率
def get_type_features(df,columns,value,operation,rename):
	if operation=="count":#统计点击次数
		add=pd.DataFrame(df.groupby(columns)[value].count()).reset_index()
	if operation=="sum":#统计转化次数
		add=pd.DataFrame(df.groupby(columns)[value].sum()).reset_index()
	if operation=="mean":#统计转化率
		add=pd.DataFrame(df.groupby(columns)[value].mean()).reset_index()
	add.columns=columns+[rename]
	df=df.merge(add,on=columns,how='left')
	return df

