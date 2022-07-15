#1读取数据
import pandas as pd
df = pd.read_excel('/Users/macbookair/Desktop/数据分析/书本配套资料及电子书/python/Python大数据分析与机器学习/源代码汇总-2020-12-16/第10章 机器学习神器：XGBoost&LightGBM模型/源代码汇总_Pycharm/信用评分卡模型.xlsx')

#2提取特征变量和目标变量
X = df.drop(columns='信用评分')
y = df['信用评分']

#3划分训练集和测试集
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X, y)

#4线性回归方程构造
print('各系数：' + str(model.coef_))
print('常数项：' + str(model.intercept_))
#此时的多元线性回归方程：y = 67.16+0.000558X1+0.162X2+0.218X3+0.00007X4-1.51X5

#5模型评估
import statsmodels.api as sm
X2 = sm.add_constant(X)  #给原来的特征变量X添加常数项，并赋给X2，这样才会有常数项
est = sm.OLS(y, X2).fit()   #进行线性方程搭建
print(est.summary())
#可以看到，模型整体的R-squared值为0.629,Adj. R-squared值为0.628,整体拟合效果一般，可能是因为数据量偏少。再来观察p值，可以发现
#大部分特征变量的P值都较小（小于0.05），的确是和目标变量'信用评分'显著相关的，而特征变量'性别'的p值达到量0.466，说明该特征变量与目标变量没有
#显著相关性，这也的确符合经验认知，因此，在多元线性回归模型中可以舍去'性别'这一特征变量
#通常以0.05为阀值，当p值小于0.05时，就认为特征变量与目标变量有显著相关性，否则就没有