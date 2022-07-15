#当有多种类别数据时，我们常常面临着对新加入的数据进行分类的问题，例如，根据口味和色泽划分新的葡萄酒的类别，根据内容形式划分新上映电影的类别
#根据过往人脸特征进行人脸识别等，这些问题都可以采用机器学习中非常经典的k近邻算法来解决。
#k近邻算法的原理：对于一个新样本，k近邻算法来的目的就是在已有数据中寻找与它最相似的K个数据，或者说'离它最近'的K个数据，
#如果这k个数据大多数属于某个类别，则该样本也属于这个类别。其计算公式：两点之间的欧式距离即｜AB|=√（X1-Y1)^2+~~~~~~+(Xn-Yn）^2

#1数据读取预处理
import pandas as pd
df = pd.read_excel('/Users/macbookair/Desktop/数据分析/书本配套资料及电子书/python/Python大数据分析与机器学习/源代码汇总-2020-12-16/第7章 K近邻算法/源代码汇总_Pycharm/葡萄酒.xlsx')
x_train = df[['酒精含量(%)','苹果酸含量(%)']]
y_train = df['分类']

#2构建模型
from sklearn.neighbors import KNeighborsClassifier as KNN
knn = KNN(n_neighbors=3)  #设置邻近参数n_neighbors也就是k值，这里设置为3，即选取最近的3个样本，如果不设置则取默认值5
knn.fit(x_train, y_train)  #传入特征变量和目标变量

#3进行预测
x_test = [[7, 1], [8, 3]]
answer = knn.predict(x_test)
print(answer)

#补充知识点：k近邻算法回归模型
#上面的代码是用k近邻算法分类模型（knn）进行分类分析，k近邻算法还可以做回归分析，对应的模型为k近邻算法回归模型（KNeighborsRregressor）
#k近邻算法分类模型将离待预测样本点最近的K个训练样本点中出现次数最多的分类作为待预测样本点点分类，
# K近邻算法回归模型则将离待预测样本点最近点k个训练样本点的平均值作为待预测样本点的分类。

from sklearn.neighbors import KNeighborsRegressor
x = [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]]
y = [1, 2, 3, 4, 5]
model = KNeighborsRegressor(n_neighbors=2)  #设置邻近参数n_neighbors也就是k值，这里设置为2，即选取最近的2个样本，如果不设置则取默认值5
model.fit(x, y)   #fit函数建立训练模型
print(model.predict([[5,5]]))    #打印预测值