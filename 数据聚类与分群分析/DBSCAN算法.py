#DBSCAN算法是一种以密度为基础的空间聚类算法，可以用密度的概念剔除不属于任何一类别的噪声点。
#该算法将簇定义为密度相连的点的最大集合，将具有足够密度的区域划分为簇，并可以发现任意形状的簇
#其基本原理1：随机选取1个未分类的样本点，2：以该样本点为圆心，按照设定的半径画圆。如果圆内的样本数大于等于设定的阀值(即设置的圆内的最小样本数），则将这些样本归为一类
#3选定圆内的所有其他样本点，按照设定的半径画圆。不断重复画圆步骤，直到没有可画的圆。将这些圆内的样本点分为一簇。
#4再次随机选择1个未分类的样本点，重复步骤2和步骤3。如果没有可画的圆，算法终止，否则重复步骤4。算法终止后如果有样本点仍未分类，则将该样本点视为离群点。。

#1读取数据
import pandas as pd
data = pd.read_excel('/Users/macbookair/Desktop/数据分析/书本配套资料及电子书/python/Python大数据分析与机器学习/源代码汇总-2020-12-16/第13章 数据聚类与分群/源代码汇总_PyCharm格式/演示数据.xlsx')

#绘制散点图
import matplotlib.pylab as plt
plt.scatter(data.iloc[:, 0], data.iloc[:, 1], c='green', marker='*')
plt.xlabel('x')
plt.ylabel('y')
plt.show()

#调用DBSCAN
from sklearn.cluster import DBSCAN
dbs = DBSCAN()   #这里不设置参数，即取默认值：画圆半径参数eps取默认值0.5,圆内最小样本数参数min_samples取默认值为5
dbs.fit(data)
label_dbs = dbs.labels_    #获取聚类结果
print(label_dbs)

#仍用散点图展示
plt.scatter(data[label_dbs == 0].iloc[:, 0], data[label_dbs == 0].iloc[:, 1], c='red', marker='o', label='class0')
plt.scatter(data[label_dbs == 1].iloc[:, 0], data[label_dbs == 1].iloc[:, 1], c='green', marker='*', label='class1')
plt.legend()   #设置图例
plt.xlabel('x')
plt.ylabel('y')
plt.show()

#KMeans算法与DBSCAN算法的对比
from sklearn.cluster import KMeans
kms = KMeans(n_clusters=2)
kms.fit(data)
label_kms = kms.labels_

#绘制散点图
plt.scatter(data[label_kms == 0].iloc[:, 0], data[label_kms == 0].iloc[:, 1], c='red', marker='o', label='class0')
plt.scatter(data[label_kms == 1].iloc[:, 0], data[label_kms == 1].iloc[:, 1], c='green', marker='*', label='class1')
plt.legend()   #设置图例
plt.xlabel('x')
plt.ylabel('y')
plt.show()

#对比之下可以看到，对于形状类似同心原单数据，KMeans算法聚类的效果较差，只能机械地将数据分为左右两部分，而无法区分外圆和内圆
#KMeans的优点：适用于常规数据集；适用于高维数据的聚类；适用与密度会发生变化的数据的聚类
#缺点：需要事先确定K值，即数据分为几类；初始中心点的选择会在较大程度上影响聚类结果；难以发现任意形状的簇

#DBSCAN算法：不需要事先确定K值；可以发现任意形状的簇；可以识别出噪声点(即离群点）；初始中心点的选择不影响聚类结果
#缺点：不适用于高维数据的聚类；不适用于密度会发生变化的数据的聚类；参数难以确定最优值
