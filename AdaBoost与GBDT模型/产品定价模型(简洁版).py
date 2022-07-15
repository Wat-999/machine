#产品定价模型
# 1.读取数据
import pandas as pd
df = pd.read_excel('/Users/macbookair/Desktop/数据分析/书本配套资料及电子书/python/Python大数据分析与机器学习/源代码汇总-2020-12-16/第9章 AdaBoost与GBDT模型/源代码汇总_Pycharm/产品定价模型.xlsx')
df.head()

df['类别'].value_counts()

df['彩印'].value_counts()

df['纸张'].value_counts()

# 2.分类型文本变量处理
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df['类别'] = le.fit_transform(df['类别'])  # 处理类别

df['类别'].value_counts()

# df['类别'] = df['类别'].replace({'办公类': 0, '技术类': 1, '教辅类': 2})
# df['类别'].value_counts()

le = LabelEncoder()
df['纸张'] = le.fit_transform(df['纸张'])

df.head()

# 3.提取特征变量和目标变量
X = df.drop(columns='价格')
y = df['价格']

# 4.划分训练集和测试集
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

# 5.模型训练及搭建
from sklearn.ensemble import GradientBoostingRegressor
model = GradientBoostingRegressor(random_state=123)
model.fit(X_train, y_train)

# 9.4.3 模型预测及评估
y_pred = model.predict(X_test)
print(y_pred)
#汇总预测值和实际值
a = pd.DataFrame()
a['预测值'] = list(y_pred)
a['实际值'] = list(y_test)
print(a.head())

#预测准确度(这个评分其实就是模型R-squared的值（即统计学中R^2))
score = model.score(X_test,y_test)
print('预测准确度：' + str(score))
#也可以用以下的代码实现，值一样
#from sklearn.metrics import r2_score
#r2 = r2_score(y_test, model.predict(X_test))

#分析数据特征的重要性，并对特征名称和特征重要性进行汇总
features = X.columns
importances = model.feature_importances_
#整理成二维表格，并按特征重要性降序排列
importances_df = pd.DataFrame()
importances_df['特征名称'] = features
importances_df['特征重要性'] = importances
importances_df.sort_values('特征重要性', ascending=False)
print(importances_df.sort_values('特征重要性', ascending=False))
