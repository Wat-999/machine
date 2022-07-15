#1、读取数据
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
df = pd.read_excel('/Users/macbookair/Desktop/数据分析/书本配套资料及电子书/python/Python大数据分析与机器学习/源代码汇总-2020-12-16/第3章 线性回归模型/源代码汇总_PyCharm格式/IT行业收入表.xlsx')#读取数据
#df.head()  #查询前5行数据
x1 = df[['工龄']]
x = np.array(x1)   #注意自变量集合一定要转换成二维结构形式，可以在这里转换，也可以在模型可视化里转换写成plt.plot(np.array(x), regr.predict(x), color='red')

y = df['薪水']

plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']    #用来正常显示中文
#plt.scatter(x, y)     #绘制散点图
plt.xlabel('工龄')     #x轴为工龄
plt.ylabel('薪资')    #y轴为薪资
#plt.show()

#2、模型搭建
from sklearn.linear_model import LinearRegression
regr = LinearRegression()
regr.fit(x, y)   #用fit（）函数完成模型搭建

#3模型预测
#y = regr.predict([[1.5], [2.5], [4.5]])
#print(y)

#4、模型可视化
plt.scatter(x, y)   #绘制散点图
plt.plot(x, regr.predict(x), color='red')  #np.array(x)函数用于创建一个数组,regr.predict（x）预测对应的因变量y，设置线为红色
plt.xlabel('工龄')   #定义x轴的列名
plt.ylabel('薪资')    #定义y轴的列名
plt.show()           #绘图

#5、线性回归方程构造
#通过coef_和intercept_属性可以得到此时趋势线的系数和截距
print('系数a：' + str(regr.coef_[0]))     #因为通过regr.coef_获得的是一个列表，所以需要通过regr.coef_【0】选取其中的元素
print('截距b：' + str(regr.intercept_))   #又因为该元素为数字，所以进行字符串拼接时需要利用str（）函数将其转换为字符串




