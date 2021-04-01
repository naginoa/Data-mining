# 阿里O2o新人赛

3800多参赛队 第50名

![Image text](https://github.com/naginoasukara/Data-mining/blob/master/image/1.png)
![Image text](https://github.com/naginoasukara/Data-mining/blob/master/image/2.png)
![Image text](https://github.com/naginoasukara/Data-mining/blob/master/image/3.png)

需要gpu加速 速率会快很多

# kaggle手写数字识别

## 1.比赛页面如下

![Image text](https://github.com/naginoasukara/Data-mining/blob/master/kaggle%E6%89%8B%E5%86%99%E6%95%B0%E5%AD%97%E8%AF%86%E5%88%AB/image/kaggle%E6%89%8B%E5%86%99%E6%95%B0%E5%AD%97%E6%AF%94%E8%B5%9B.png)

## 2.代码迭代次数及时间

![Image text](https://github.com/naginoasukara/Data-mining/blob/master/kaggle%E6%89%8B%E5%86%99%E6%95%B0%E5%AD%97%E8%AF%86%E5%88%AB/image/%E8%BF%AD%E4%BB%A3%E6%AC%A1%E6%95%B0%E5%8F%8A%E6%97%B6%E9%97%B4.png)

## 3.xgb图

![Image text](https://github.com/naginoasukara/Data-mining/blob/master/kaggle%E6%89%8B%E5%86%99%E6%95%B0%E5%AD%97%E8%AF%86%E5%88%AB/image/xgb_importance.png)

## 4.排名及结果

![Image text](https://github.com/naginoasukara/Data-mining/blob/master/kaggle%E6%89%8B%E5%86%99%E6%95%B0%E5%AD%97%E8%AF%86%E5%88%AB/image/kaggle%E6%8E%92%E5%90%8D%E5%8F%8A%E5%BE%97%E5%88%86.png)

# tf衣服图片识别率提升
tf官方教程的识别网络结构是一开始全部对图片碾平成一维, 之后过两个全连接层.

我的改进是首先将数据reshape成四维, [batch, h,w,c]. CNN1+dropout+CNN2  256全连接 dropout 0.2 128全连接 68全连接 dropout0.2 10全连接

准确率可以提升从0.86提升到0.91
