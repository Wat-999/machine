#GBDT是Gradient Boosting Decision Tree(梯度提升树）。GBDT算法也是一种非常实用的Boosting算法，它与AdaBoosting算法
#它与AdaBoosting算法的区别在于：AdaBoosting算法根据分类效果调整权重不断迭代，最终生成强学习器；
#GBDT算法则将损失函数的负梯度作为残差的近似值，不断使用残差迭代和拟合回归树，最终生成强学习器。简单来说，AdaBoost算法是调整权重，而GBDT算法是拟合残差
#GBDT算法既能做分类分析，也能做回归分析，对应的模型分别为GBDT分类模型（GradientBoostingClassifier）和GBDT回归模型（GradientBoostingRegressor）
#其弱学习器对应的都是相应的分类决策树模型和回归决策树模型

#GBDT分类模型
from sklearn.ensemble import GradientBoostingClassifier
x = [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]]
y = [0, 0, 0, 1, 1]
model = GradientBoostingClassifier(random_state=123)
model.fit(x, y)
print(model.predict([[5, 5]]))

#GBDT回归模型
from sklearn.ensemble import GradientBoostingRegressor
x = [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]]
y = [1, 2, 3, 4, 5]
model = GradientBoostingRegressor(random_state=123)
model.fit(x, y)
print(model.predict([[5, 5]]))