#XGBoost算法在某种程度上可以说是GBDT版算法的改良版，两者在本质上都市利用了Boosting算法中拟合残差的思想，在信用卡额度预测模型，
#其中初始决策树的预测结果不完全准确，会产生一些残差，因此会用新的决策树来拟合残差，新的决策树又会产生新的残差，这时再构造新的决策树来拟合新的残差～～～
#如此迭代下去，直至符合预先设定的条件为止
#作为对GBDT算法的高效实现，XGBoost算法在以下两方面进行两优化
#算法本身的优化：XGBoot算法的损失函数，除两本身的损失，还加上了正则化部分，防止过拟合，泛化能力更强。XGBoost算法的损失函数是对误差部分做二阶泰勒展开，更加准确
#算法运行效率的优化：对每个弱学习器，如决策树建立的过程做并行选择，找到合适的子节点分裂特征和特征值，从而提升运行效率
#XGBoost算法既能做分类分析，又能做回归分析，对应的模型分别为XGBoost分类模型（XGBClassifier）和XGBoost回归模型（XGBRegressor）

#XGBoost分类模型
from xgboost import XGBClassifier
import numpy as np
#XGBoost模型对特征变量只支持array数组类型或Dataframe二维表格类型，所以这里使用Numpy库对array()函数将list列表类型的数据转换为array数组类型的数据
x = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]])
y = [0, 0, 0, 1, 1]
model = XGBClassifier()
model.fit(x, y)
print(model.predict(np.array([[5, 5]])))


#XGBoost回归模型
from xgboost import XGBRegressor
import numpy as np
x = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]])
y = [1, 2, 3, 4, 5]
model = XGBRegressor()
model.fit(x, y)
print(model.predict(np.array([[5, 5]])))


