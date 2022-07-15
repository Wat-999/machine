import matplotlib.pyplot as plt
x = [[1], [2], [4], [5]]
y = [2, 4, 6, 8]
#plt.scatter(x, y)  #绘制散点图
#plt.show()

#2引入sklearn-Learn库搭建模型  y=ax+b
from sklearn.linear_model import LinearRegression  #从sklearn-Learn库引入线性回归的相关模块linearRegression
regr = LinearRegression()      #构造初始的线性回归模型并命名为regr
regr.fit(x, y)                 #用fit（）函数完成模型搭建，此时regr就是一个搭建好的模型

#3模型预测
#用搭建好的regr来预测数据，假设自变量是1.5，那么使用predict（）函数就能预测对应的应变量y
#y = regr.predict([[1.5]])  #注意自变量还是要写成【【】】二维结构模式，获得的预测结果y为一个一维数组
#print(y)    #打印预测值

#预测多个自变量
#y = regr.predict([[1.5], [2.5], [4.5]])
#print(y)

#4.模型可视化
plt.scatter(x, y)
plt.plot(x, regr.predict(x))
plt.show()

#5.线性回归方程构造
#通过coef_和intercept_属性可以得到此时趋势线的系数和截距
print('系数a：' + str(regr.coef_[0]))     #因为通过regr.coef_获得的是一个列表，所以需要通过regr.coef_【0】选取其中的元素
print('截距b：' + str(regr.intercept_))   #又因为该元素为数字，所以进行字符串拼接时需要利用str（）函数将其转换为字符串

