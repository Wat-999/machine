#线性回归是用于对连续变量进行预测，如预测收入范围、客户价值等
#1.读取数据
import pandas as pd
df = pd.read_excel('/Users/macbookair/Desktop/数据分析/书本配套资料及电子书/python/Python大数据分析与机器学习/源代码汇总-2020-12-16/第3章 线性回归模型/源代码汇总_PyCharm格式/客户价值数据表.xlsx')
print(df.head())  #打印前5行
x = df[['历史贷款金额', '贷款次数', '学历', '月收入', '性别']]
y = df['客户价值']

#2模型搭建（多元线性回归模型）y=k+k1x1+k2x2+k3x3+~~~~knxn
from sklearn.linear_model import LinearRegression  #从sklearn-Learn库引入线性回归的相关模块linearRegression
regr = LinearRegression()     #构造初始的线性回归模型并命名为regr
regr.fit(x, y)      #完成模型搭建

#3线性回归方程构造
print('各系数：' + str(regr.coef_))   #获取系数k1～kn
print('常数项：' + str(regr.intercept_))  #获取常数项k
#y=-208+0.057x1+96x2+113x3+0.056x4+1.9x5    注意各系数中每个系数后面三位代表左右移动小数点几位
#4模型评估
import statsmodels.api as sm  #引入用于评估线性回归模型的statsmodels库
x2 = sm.add_constant(x)      #用add_constant(x)函数给原来的特征变量x添加常数项，并赋给x2，这样才有多元线性表达式中的常数项，即常数项k
est = sm.OLS(y, x2).fit()  #用OLS(y, x2)和fit()函数对y和x2进行线性回归方程搭建，
print(est.summary())       #在jupyter notebook中可以直接写est.summary（）

##对于打印出来的内容进行模型评估而言，通常需要关心图中的R—squared、Adj.R—squared和P值信息。
#这里的R—squared为0.571，Adj.R—squared为0.553，说明模型整体的线性拟合效果不是特别好，
# 可能是因为本案列的数据量偏少，不过在此数据量条件下也算可以接受的结果
#观察p值，可以发现大部分特征变量的p值都较小，的确与目标变量（即客户价值）显著相关，
#而"性别"这一特征变量的p值达到量0.951，即与目标变量没有显著相关性，这个结论也符合经验认知，因此在之后的建模中可以舍去性别这一特征变量
