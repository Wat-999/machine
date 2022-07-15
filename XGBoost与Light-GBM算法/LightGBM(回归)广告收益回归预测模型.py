#1数据读取
import pandas as pd
df = pd.read_excel('/Users/macbookair/Desktop/数据分析/书本配套资料及电子书/python/Python大数据分析与机器学习/源代码汇总-2020-12-16/第10章 机器学习神器：XGBoost&LightGBM模型/源代码汇总_Pycharm/广告收益数据.xlsx')

#2提取特征变量和目标变量
X = df.drop(columns='收益')
y = df['收益']

#3划分训练集和测试集
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

#4模型训练及搭建
from lightgbm import LGBMRegressor
model = LGBMRegressor()
model.fit(X_train, y_train)

#5模型预测及评估
#对测试集数据进行预测
y_pred = model.predict(X_test)
print(y_pred)

#汇总预测值和实际值
a = pd.DataFrame()
a['预测值'] = list(y_pred)
a['实际值'] = list(y_test)
#print(a.head())

#手动输入预测
#X = [[71, 11, 2]]
#print(model.predict(X))

#查看R-squared值评价模型的拟合效果(因为LGBMRegressor是个回归模型）
from sklearn.metrics import r2_score
r2 = r2_score(y_test, model.predict(X_test))
print('R-squared值:' + str(r2))
#print(model.score(X_test, y_test))  #这是LGBMRegressor自带的score()函数效果等同上面

#6分析数据特征的重要性，并对特征名称和特征重要性进行汇总
features = X.columns  # 获取特征名称
importances = model.feature_importances_  # 获取特征重要性
importances_df = pd.DataFrame()
importances_df['特征名称'] = features
importances_df['特征重要性'] = importances
importances_df.sort_values('特征重要性', ascending=False)
print(importances_df.sort_values('特征重要性', ascending=False))

#7模型参数调优
from sklearn.model_selection import GridSearchCV
parameters = {'num_leaves': [15, 31, 62], 'n_estimators': [20, 30, 50, 70], 'learning_rate': [0.1, 0.2, 0.3, 0.4]}  #模型待调优参数的候选值范围
#num_leaves:决策树的最大叶子节点数，即决策树最多有多少个叶子节点，默认取值31。
#因为LightGBM模型使用的是leaf-wise生长策略，所以在调节树的复杂度时常用的参数是num_leaves,而不是树的最大深度参数max_derth
#n_estimators:弱学习器的个数，或者说是弱学习器的最大迭代次数
#learning_rate：学习率，又称为每个弱学习器的权重缩减系数，取值范围(0,1],默认取0.1。取较小值意味着要达到一定的误分类数或学习效果，需要更多迭代次数和更多学习器
model = LGBMRegressor()  #构建LGBMClassifier模型
grid_search = GridSearchCV(model, parameters, scoring='r2', cv=5)#将模型和待调优的参数候选值范围传入，模型的评估标准，交叉验证次数
grid_search.fit(X_train, y_train)   #传入训练集数据
print(grid_search.best_params_)     #输出参数最优值 {'learning_rate': 0.3, 'n_estimators': 50, 'num_leaves': 31}

#获得参数最优值重新搭建模型
model = LGBMRegressor(n_estimators=50, num_leaves=31, learning_rate=0.3)
model.fit(X_train, y_train)

#查看新模型的AUC值
from sklearn.metrics import r2_score
r2 = r2_score(y_test, model.predict(X_test))
print('新模型：' + str(r2))

#总结XGBoost模型和LightGBM模型目前是非常流行等俩个机器学习，在各种数据挖掘竞赛中经常大放异彩，无论是在回归分析还是分类分析中，
#两个模型都能达到较为理想等效果