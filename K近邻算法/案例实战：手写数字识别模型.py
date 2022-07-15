#1读取数据
import pandas as pd
df = pd.read_excel('/Users/macbookair/Desktop/数据分析/书本配套资料及电子书/python/Python大数据分析与机器学习/源代码汇总-2020-12-16/第7章 K近邻算法/源代码汇总_Pycharm/手写字体识别.xlsx')

#提取特征变量和目标变量
X = df.drop(columns='对应数字')
y = df['对应数字']
#所有样本的1*1024矩阵都由0和1构成，故无须标准处理，
#如果在其他场景中出现数量级相差较大的特征变量，则需要对数据标准化处理，如下
#from sklearn.preprocessing import StandardScaler
#x = StandardScaler().fit_transform(x)

#3划分训练集和测试集
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

#4模型搭建
from sklearn.neighbors import KNeighborsClassifier as KNN
knn = KNN(n_neighbors=5)
knn.fit(X_train,y_train)

#5模型预测与评估
y_pred = knn.predict(X_test)    #对测试集数据进行预测

#将预测值与实际值进行对比
a = pd.DataFrame()
a['预测值'] = list(y_pred)
a['实际值'] = list(y_test)
print(a.head())

#查看测试集的预测准确度
from sklearn.metrics import accuracy_score
score = accuracy_score(y_pred, y_test)
#score = knn.score(X_test, y_test)   #k近邻算法分类模型自带评分功能
print(score)



#参数调优
import numpy as np
from sklearn.model_selection import GridSearchCV
parameters = {'n_neighbors': np.arange(1, 10, 1)}    #参数候选值范围np.arange()构造1～9范围内，间隔为1的整数数据集
knn = KNN()
grid_search = GridSearchCV(knn, parameters, cv=5)  #5折交叉验证
grid_search.fit(X_train, y_train)  #以准确度为基础进行网格搜索，寻找参数最优值
grid_search.best_params_['n_neighbors']     #获取参数最优值
print(grid_search.best_params_['n_neighbors'])

#4单参数调优后模型搭建
from sklearn.neighbors import KNeighborsClassifier as KNN
knn = KNN(n_neighbors=1)
knn.fit(X_train,y_train)

#5模型预测与评估
y_pred = knn.predict(X_test)    #对测试集数据进行预测

#将预测值与实际值进行对比
a = pd.DataFrame()
a['预测值'] = list(y_pred)
a['实际值'] = list(y_test)
print(a.head())

#查看测试集的预测准确度
from sklearn.metrics import accuracy_score
score = accuracy_score(y_pred, y_test)
#score = knn.score(X_test, y_test)   #k近邻算法分类模型自带评分功能
print(score)
