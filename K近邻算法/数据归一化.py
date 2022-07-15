# # 2.数据归一化代码演示
# 2.1 min-max标准化
import pandas as pd
df = pd.read_excel('/Users/macbookair/Desktop/数据分析/书本配套资料及电子书/python/Python大数据分析与机器学习/源代码汇总-2020-12-16/第7章 K近邻算法/源代码汇总_Pycharm/葡萄酒2.xlsx')
X = df[['酒精含量(%)','苹果酸含量(%)']]
y = df['分类']

from sklearn.preprocessing import MinMaxScaler
X_new = MinMaxScaler().fit_transform(X)
X_new

# 2.2 Z-score标准化
from sklearn.preprocessing import StandardScaler
X_new = StandardScaler().fit_transform(X)
X_new