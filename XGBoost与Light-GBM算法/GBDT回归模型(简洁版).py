# 1.读取数据
import pandas as pd
df = pd.read_excel('/Users/macbookair/Desktop/数据分析/书本配套资料及电子书/python/Python大数据分析与机器学习/源代码汇总-2020-12-16/第10章 机器学习神器：XGBoost&LightGBM模型/源代码汇总_Pycharm/信用评分卡模型.xlsx')

# 2.提取特征变量和目标变量
X = df.drop(columns='信用评分')
y = df['信用评分']

# 3.划分测试集和训练集
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

# 4.模型训练及搭建
from xgboost import XGBRegressor
model = XGBRegressor()
model.fit(X_train, y_train)

# 5.模型预测及评估
y_pred = model.predict(X_test)
print(y_pred)

a = pd.DataFrame()  # 创建一个空DataFrame
a['预测值'] = list(y_pred)
a['实际值'] = list(y_test)
a.head()

from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_pred)
print(r2)

# 6.查看特征重要性
features = X.columns
importances = model.feature_importances_
#整理成二维表格，并按特征重要性降序排列
importances_df = pd.DataFrame()
importances_df['特征名称'] = features
importances_df['特征重要性'] = importances
importances_df.sort_values('特征重要性', ascending=False)
print(importances_df.sort_values('特征重要性', ascending=False))

# **补充知识点1：XGBoost回归模型的参数调优**
from sklearn.model_selection import GridSearchCV
parameters = {'max_depth': [1, 3, 5], 'n_estimators': [50, 100, 150], 'learning_rate': [0.01, 0.05, 0.1, 0.2]}
clf = XGBRegressor()  # 构建回归模型
grid_search = GridSearchCV(model, parameters, scoring='r2', cv=5)

grid_search.fit(X_train, y_train)
print(grid_search.best_params_ )    # {'learning_rate': 0.1, 'max_depth': 3, 'n_estimators': 50}



#用参数的最优值搭建新模型
model = XGBRegressor(max_depth=3, n_estimators=50, learning_rate=0.1)
model.fit(X_train, y_train)
#进行调优后模型评估
from sklearn.metrics import r2_score
r2 = r2_score(y_test, model.predict(X_test))
print('调优后R-squared：' + str(r2))


