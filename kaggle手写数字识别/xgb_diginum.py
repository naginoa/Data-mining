import xgboost as xgb
import pandas as pd
import time
import numpy as np
import matplotlib as plt


now = time.time()

dataset = pd.read_csv("data/train.csv")

train = dataset.iloc[:,1:].values
labels = dataset.iloc[:,:1].values

tests = pd.read_csv("data/test.csv")
#test_id = range(len(tests))
test = tests.iloc[:,:].values


params={
'booster':'gbtree',
# 这里手写数字是0-9，是一个多类的问题，因此采用了multisoft多分类器，
'objective': 'multi:softmax',
'num_class':10, # 类数，与 multisoftmax 并用
'gamma':0.05,  # 在树的叶子节点下一个分区的最小损失，越大算法模型越保守 。[0:]
'max_depth':12, # 构建树的深度 [1:]
#'lambda':450,  # L2 正则项权重
'subsample':0.4, # 采样训练数据，设置为0.5，随机选择一般的数据实例 (0:1]
'colsample_bytree':0.7, # 构建树树时的采样比率 (0:1]
#'min_child_weight':12, # 节点的最少特征数
'silent':1 ,
'eta': 0.005, # 如同学习率
'seed':710,
'nthread':4,# cpu 线程数,根据自己U的个数适当调整
}

plst = list(params.items())

#Using 10000 rows for early stopping.
offset = 35000  # 训练集中数据50000，划分35000用作训练，15000用作验证

num_rounds = 30000 # 迭代次数
xgtest = xgb.DMatrix(test)

# 划分训练集与验证集
xgtrain = xgb.DMatrix(train[:offset,:], label=labels[:offset])
xgval = xgb.DMatrix(train[offset:,:], label=labels[offset:])

# return 训练和验证的错误率
watchlist = [(xgtrain, 'train'),(xgval, 'val')]


# training model
# early_stopping_rounds 当设置的迭代次数较大时，early_stopping_rounds 可在一定的迭代次数内准确率没有提升就停止训练
model = xgb.train(plst, xgtrain, num_rounds, watchlist,early_stopping_rounds=100)
# 用于存储训练出的模型
model.save_model('model/xgb.model')
# 转存模型
model.dump_model('model/0p-dump.raw.txt')

preds = model.predict(xgtest,ntree_limit=model.best_iteration)


# 将预测结果写入文件
np.savetxt('submission_xgb_MultiSoftmax.csv',np.c_[range(1,len(test)+1),preds],
                delimiter=',',header='ImageId,Label',comments='',fmt='%d')


cost_time = time.time()-now
print("end ......",'\n',"cost time:",cost_time,"(s)......")
#制图
xgb.plot_importance(model)
#xgb.plot_tree(model, num_trees=2)