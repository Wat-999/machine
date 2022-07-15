#AdaBoost算法的核心思想
#AdaBoost算法是一种有效而实用的Boosting算法，它以一种高度高度自适应的方式按顺序训练弱学习器。针对分类问题，AdaBoost算法根据前一次的分类效果
#调整数据的权重，在上一个弱学习器中分类错误的样本的权重会在下一个弱学习器中增加，分类正确的样本的权重则相应减少，并且在每一轮迭代时会向模型加入
#新的弱学习器，不断重复调整权重和训练学习器，直到误分类数低于预设值或迭代次数达到指定最大值，
#最终得到一个强学习器。简单来说，AdaBoost算法的核心思想就是调整错误样本的权重，进行迭代升级。
#AdaBoost算法既能做分类分析，也能做回归分析，对应的模型分别为AdaBoost分类模型（AdaBoostClassifier）和AdaBoost回归模型（AdaBoostRegressor）
#其弱学习器对应的都是相应的分类决策树模型和回归决策树模型

#AdaBoost分类模型
from sklearn.ensemble import AdaBoostClassifier
x = [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]]
y = [0, 0, 0, 1, 1]
model = AdaBoostClassifier(random_state=1)
model.fit(x, y)
print(model.predict([[5, 5]]))


#AdaBoos回归模型
from sklearn.ensemble import AdaBoostRegressor
x = [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]]
y = [1, 2, 3, 4, 5]
model = AdaBoostRegressor(random_state=1)
model.fit(x, y)
print(model.predict([[5, 5]]))
