#数据标准化(也称为数据归一化),它的主要目的是消除不同特征变量量纲级别相差太大造成的不利影响。对于以特征处理为算法基础的机器学习算法(如K近邻算法)
#数据标准化尤为重要

#min-max标准化也称离差标准化，它利用原始数据的最大值和最小值把原始数据转换到[0,1]区间内 X*=(x-min)/(max-min)
#其中x、X*分别为转换前和转换后的值，max、min分别为原始数据的最大值和最小值。例如一个样本集中最大值为100，最小值为40，若此时x为50
#则min-max标准化后的值：X*=(50-40)/(100-40)=0.167

#构造演示数据
import pandas as pd
X = pd.DataFrame({'酒精含量(%)': [50, 60, 40, 80, 100], '苹果酸含量(%)': [2, 1, 1, 3, 2]})
y= [0, 0, 0, 1, 1]

#这里只需要对特征变量X进行处理，目标变量y不用处理，在python中可以直接调用min-max标准化的相关模块
from sklearn.preprocessing import MinMaxScaler   #引入相关模块minmaxscaler
X_new = MinMaxScaler().fit_transform(X)          #用fit_transform(X)函数对原始数据进行min-max标准化
#在实际应用中，通常将所有数据标准化后，再进行训练集和测试集划分
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_new, y, test_size=0.2, random_state=123)


#Z-score标准化
#Z-score标准化也称均值归一化，通过原始数据的均值(mean)和标准差(std)对数据进行标准化。标准化后的数据符合标准正态分布，即均值为0，标准差为1
#X*=(X-mean）/std 其中x和X*分别为转换前和转换后的值，mean为原始数据的均值，std为原始数据的标准差
from sklearn.preprocessing import StandardScaler   #引入相关模块StandardScaler
X_new = StandardScaler().fit_transform(X)  #fit_transform(X)对原始数据进行Z-score标准化
print(X_new)
#其中第一列为"酒精含量"标准化后的值，第二列为"苹果酸含量"标准化后的值，此时他们是均值为0、标准差为1的标准正态分布
#总结来说，数据标准化并不复杂，两三行代码就能避免很多问题，因此，对一些量纲相差较大的特征变量，实战中通常会先进行数据标准化，再进行训练集和测试集划分

#除列k近邻算法模型，还有一些模型也是基于距离的，所以量纲对模型影响较大，就需要进行数据标准化，如支持向量机模型、KMeans聚类分析、PCA（主成分分析）等
#此外对于一些线性模型，线性回归模型、逻辑回归模型，有时也需要进行数据标准化处理

#对于树模型则无须做数据标准化处理，因为数值缩放不影响分裂点位置，对树模型的结构不造成影响。因此，决策树模型及基于决策树模型的随机森林模型、
#AdaBoost模型、GBDT模型、XGBoost模型、LightGBM模型通常都不需要数据标准化处理，因为他们不关心变量的值，而是关心变量的分布和变量之间的条件概率
#在树模型相关的机器学习模型中，进行数据标准化对预测结果不会产生影响

#在实际工作中，如果不确定是否要做数据标准化，可以先尝试做一做数据标准化，看看模型预测准确度是否有提升，如果提升较明显，则推荐进行数据标准化。
