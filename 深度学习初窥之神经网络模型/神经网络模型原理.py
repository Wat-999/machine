# # 1.神经网络模型简单代码实现(分类）
X = [[1, 0], [5, 1], [6, 4], [4, 2], [3, 2]]
y = [0, 1, 1, 0, 0]
from sklearn.neural_network import MLPClassifier
mlp = MLPClassifier()
mlp.fit(X, y)     #这里直接使用训练集作为测试集构建模型
y_pred = mlp.predict(X)

import pandas as pd
a = pd.DataFrame()  # 创建一个空DataFrame
a['预测值'] = list(y_pred)
a['实际值'] = list(y)
print(a)

# 补充知识点 - 神经网络回归模型：MLPRegressor
from sklearn.neural_network import MLPRegressor
X = [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]]
y = [1, 2, 3, 4, 5]
model = MLPRegressor(random_state=123)  # 设置random_state随机状态参数，使得每次训练的模型都是一样的
model.fit(X, y)
print(model.predict([[5, 5]]))