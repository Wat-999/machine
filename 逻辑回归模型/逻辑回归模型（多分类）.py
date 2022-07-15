#构造数据，此时y有3个分类-1，0，1
x = [[1, 0], [5, 1], [6, 4], [4, 2], [3, 2]]
y = [-1, 0, 1, 1, 1]
#2、使用逻辑回归模型进行拟合
from sklearn.linear_model import LogisticRegression  #引入逻辑回归模型 LogisticRegression
model = LogisticRegression()       #将逻辑回归模型赋给变量model
model.fit(x, y)                    #用fit（）进行模型的训练

##3模型预测
#model.predict(x)  #直接将其赋值和下面演示的多个数据是一样的
#print(model.predict(x))
model.predict([[0, 0]])    #用predict（）函数进行预测，这里写俩组中括号，因为predict函数默认接收一个多维的数据，将其用于同时预测多个数据时容易理解
print(model.predict([[0, 0]]))  #打印预测结果

#4预测概率（逻辑回归模型的本质是预测概率，而不是直接预测具体类别（如属于0还是1））
#y_pred_proba = model.predict_proba(x)  #可以直接将y_pred_proba打印出来，它是一个NumPy格式的二维数组、、直接将其赋值和下面演示的多个数据是一样的
y_pred_proba = model.predict_proba([[0, 0]])
import pandas as pd
a = pd.DataFrame(y_pred_proba, columns=['分类为-1的概率', '分类为0的概率', '分类为1的概率'])  #用二维数组创建DataFrame，打印出来更美观
print(a)  #打印结果

#可以看到预测分类为-1的概率最大，因此数据【0，0】被预测为分类-1