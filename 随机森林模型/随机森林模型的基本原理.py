#随机森林（Random Forest）是一种经典的Bagging模型，其弱学习器为决策树模型。
#随机森林模型会在原始数据集中随机抽样，构成n个不同的样本数据集，然后根据这些数据集搭建n个不同的决策树模型，
#最后根据这些决策树模型的平均值（针对回归模型），或者投票情况（针对分类模型）来获取最终结果。
#为了保证模型的泛化能力（或者说通用能力），随机森林模型在建立每棵树时，往往会遵循'数据随机'和'特征随机'这两个基本原则
#1数据随机
#从所有数据当中有放回地随机抽取数据作为其中一个决策树模型的训练数据，例如有1000个数据，有放回地抽取1000次，构成一组新的数据，用于训练某一个决策树模型

#2特征随机
#如果每个样本的特征维度为M，指定一个常数k<M，随机地从M个特征中选取k个特征，在使用python构造随机森林模型时，默认选取特征的个数k为√M.
#与单独的决策树模型相比，随机森林模型由于集成了多个决策树，其预测结果会更准确，且不容易造成过拟合现象，泛化能力更强

#随机森林模型的代码实现
#和决策树模型一样，随机森林模型既能进行分类分析，又能进行回归分析，对应的模型分别为随机森林分类模型（RandomForestClassifier）和
#随机森林回归模型（RandomForestRegressor），随机森林分类模型的弱学习器是分类决策树模型，随机森林回归模型的弱学习器则是回归决策树模型。


#随机森林分类模型
from sklearn.ensemble import RandomForestClassifier
X = [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]]
y = [0, 0, 0, 1, 1]

model = RandomForestClassifier(n_estimators=10, random_state=123)    #n_estimators=10即共有10个决策树模型作为弱学习器
model.fit(X, y)

print('随机森林分类模型预测：' + str(model.predict([[5, 5]])))


#随机森林回归模型
from sklearn.ensemble import RandomForestRegressor
X = [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]]
y = [1, 2, 3, 4, 5]

model = RandomForestRegressor(n_estimators=10, random_state=123)
model.fit(X, y)

print('随机森林回归模型预测：' + str(model.predict([[5, 5]])))
