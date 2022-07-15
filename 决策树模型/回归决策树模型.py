#决策树除了能进行分类分析，还能进行回归分析，即预测连续变量，此时的决策树称为回归决策树
#回归决策树模型的概念和分类决策树模型基本一致，最大的不同就是其划分标准不是基尼系数或信息熵，而是均方误差MSE
from sklearn.tree import DecisionTreeRegressor  #引入回归决策树模型DecisionTreeRegressor
x = [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]]    #x是特征变量，共有5个训练数据，每个数据有2个特征，如数据[1,2],它的第一个特征的数值为1，第二个特征的数值为2
y = [1, 2, 3, 4, 5]     #y是目标变量，为一个连续变量
model = DecisionTreeRegressor(max_depth=2, random_state=0)    #引入模型设置决策树最大深度参数max_depth=2，随机状态参数random_state=0为0，这里0没有意义可以换成其他数字，它是一个种子参数，可使每次运行结果一致
model.fit(x, y)   #用fit（）函数训练模型
print(model.predict([[9, 9]]))   #用predict（）函数对数据[9,9]进行预测，预测拟合值为4.5