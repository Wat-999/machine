#1、读取数据
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
df = pd.read_excel('/Users/macbookair/Desktop/数据分析/书本配套资料及电子书/python/Python大数据分析与机器学习/源代码汇总-2020-12-16/第3章 线性回归模型/源代码汇总_PyCharm格式/IT行业收入表.xlsx')#读取数据
#df.head()  #查询前5行数据
x1 = df[['工龄']]
x = np.array(x1)   #注意自变量集合一定要转换成二维结构形式，可以在这里转换，也可以在模型可视化里转换写成plt.plot(np.array(x), regr.predict(x), color='red')
y = df['薪水']

#2、模型搭建（一元二次线性回归模型） y=ax^2+bx+c
from sklearn.linear_model import LinearRegression   #从sklearn-Learn库引入线性回归的相关模块linearRegression
from sklearn.preprocessing import PolynomialFeatures #引入用于增加一个多次项内容的模块PolynomialFeatures
poly_reg = PolynomialFeatures(degree=2)              #设置最高次项(degree=2)即x^2为二次项，为生成二次项数据（x^2）做准备
x_ = poly_reg.fit_transform(x)   #将原有的x转换为一个新的二维数组x_，该二维数组包含新生成的二次项数据（x^2）和原有的一次项数据（x）
regr = LinearRegression()   #构造初始的线性回归模型并命名为regr
regr.fit(x_, y)   #用fit（）函数完成模型搭建

#3模型预测
#y = regr.predict([[1.5], [2.5], [4.5]])
#print(y)

#4、模型可视化

plt.scatter(x, y)
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']    #用来正常显示中文
plt.plot(x, regr.predict(x_), color='red', label='y = 400x^2-743.68x+13988')   # #np.array(x)函数用于创建一个数组,regr.predict（x）预测对应的因变量y，color设置线为颜色，label设置标签
plt.legend(loc='upper left')  #设置图例位置
plt.xlabel('工龄')   #定义x轴的列名
plt.ylabel('薪资')    #定义y轴的列名
plt.title('IT行业薪水')  #设置标题
plt.show()
#5、一元二次线性回归方程构造
print(regr.coef_)         #获取系数a、b
print(regr.intercept_)    #获取常数项c
#[   0.         -743.68080444  400.80398224]
#13988.159332096888
#打印出来的结果：第一行为系数，有3个数：第一个数0对应x_中常数项的系数，这也是为什么之前说x_的常数项不会对分析结果产生影响；
#第2个数对应x_中一次项（x）的系数，即系数b；
#第3个数对应x_中二次项（x^2）的系数，即系数a；
#第二行的数对应常数项c，因此拟合得到的一元二次线性回归方程为y = 400x^2-743.68x+13988


#6线性回归模型评估
import statsmodels.api as sm  #引入用于评估线性回归模型的statsmodels库
x2 = sm.add_constant(x_)      #用add_constant(x_)函数给原来的特征变量x添加常数项，并赋给x2，这样才有y=ax^2+bx+c中的常数项，即常数项c
est = sm.OLS(y, x2).fit()    #用OLS(y, x2)和fit()函数对y和x2进行线性回归方程搭建，
print(est.summary())         #在jupyter notebook中可以直接写est.summary（）

#补充评估规则：R—squared和Adj.R—squared的取值范围为0～1，它们的值越接近1，则模型的拟合程度越高；
#P值在本质上是个概率值，其取值范围也为0～1，P值越接近0，则特征变量的显著性越高，即该特征变量真的和目标变量具有相关性

#对于打印出来的内容进行模型评估而言，通常需要关心图中的R—squared、Adj.R—squared和P值信息。
#这里的R—squared为0.931，Adj.R—squared为0.930，说明模型的线性拟合程度较高；
#这里的P值有三个常数项const和x2的P值都约等于0，x1的p值约等于0.023，所以这三个变量都和目标变量（薪水）显著相关，即真的具有相关性，而不是偶然因素导致的
