#线性回归模型是一种回归模型，它用于对连续变量进行预测，如预测收入范围、客户价值等。 其取值范围（-∞，+∞）
#逻辑回归模型也简称分类模型，它用于对离散变量进行预测，本质上预测的是类别的的概率，其取值范围（0，1）
#分类模型与回归模型的区别在于其预测的变量不是连续的，而是离散的一些类别，
# 例如，最常见的二分类模型可预测一个人是否会违约、客户是否会流失、肿瘤是良性还是恶性等
#本案列处理二分类问题
#1、构造数据
x = [[1, 0], [5, 1], [6, 4], [4, 2], [3, 2]]
y = [0, 1, 1, 0, 0]
#2、使用逻辑回归模型进行拟合
from sklearn.linear_model import LogisticRegression  #引入逻辑回归模型 LogisticRegression
model = LogisticRegression()       #将逻辑回归模型赋给变量model
model.fit(x, y)                    #用fit（）进行模型的训练
#3模型预测
model.predict(x)  #直接将其赋值和下面演示的多个数据是一样的
print(model.predict(x))
#model.predict([[2, 2]])    #用predict（）函数进行预测，这里写俩组中括号，因为predict函数默认接收一个多维的数据，将其用于同时预测多个数据时容易理解
#print(model.predict([[2, 2]]))  #打印预测结果

#4预测概率（逻辑回归模型的本质是预测概率，而不是直接预测具体类别（如属于0还是1））
y_pred_proba = model.predict_proba(x)  #可以直接将y_pred_proba打印出来，它是一个NumPy格式的二维数组
import pandas as pd
a = pd.DataFrame(y_pred_proba, columns=['分类为0的概率', '分类为1的概率'])  #用二维数组创建DataFrame，打印出来更美观
print(a)
#可以看到第1、4、5行中，预测分类为0的概率大于预测分类为1的概率，因此他们将被预测为分类0；
#第2、3行中，预测分类为1的概率大于预测分类为0的概率，因此他们将被预测为分类1
#最终和print(a)打印的结果相符

#5获取截距K0及系数k1和k2
print(model.coef_)      #打印输出系数k1和k2
print(model.intercept_)  #打印输出截距项k0
#本案列因为只有俩个特征变量，所以逻辑回归计算概率的数学原理可以表示为如下所示的公式。
#现在需要计算截距项k0及系数k1和k2，使得预测的概率尽可能准确。
#注意在二分类模型（0和1俩个分类）中，该P值（本案列）默认预测的是分类为1（或者说二分类中数值较大的分类）的概率

#6批量查看预测概率
import numpy as np
for i in range(5):   #这里有5条数据，所以循环5次
    print(1/(1+np.exp(-(np.dot(x[i], model.coef_.T) + model.intercept_))))  #预测 y=1（1类别）的概率
#np.exp（）用于进行指数运算（即e^x)
#np.dot()用于进行数据点乘运算，即将系数和特征值一一相乘
#model.coef_.T中.T则是将数据进行转置，为点乘运算做准备