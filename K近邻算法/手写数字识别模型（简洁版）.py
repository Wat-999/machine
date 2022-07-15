#主要分为3步：第一步训练模型，第二步处理图片，第三步进行预测
#1训练模型
#（1）读取数据
import pandas as pd
df = pd.read_excel('/Users/macbookair/Desktop/数据分析/书本配套资料及电子书/python/Python大数据分析与机器学习/源代码汇总-2020-12-16/第7章 K近邻算法/源代码汇总_Pycharm/手写字体识别.xlsx')

#(2)提取特征变量和目标变量
X = df.drop(columns='对应数字')
y = df['对应数字']

#（3）划分训练集与测试集
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

#(4)搭建并训练模型
from sklearn.neighbors import KNeighborsClassifier as KNN
knn = KNN(n_neighbors=5)  #设定选取k值即选取最近的样本个数
knn.fit(X_train, y_train)

#2处理图片
#（1）图片读取、大小调整、灰度处理
from PIL import Image
img = Image.open('/Users/macbookair/Desktop/数据分析/书本配套资料及电子书/python/Python大数据分析与机器学习/源代码汇总-2020-12-16/第7章 K近邻算法/源代码汇总_Pycharm/测试图片.png')
img = img.resize((32, 32)) #大小调整
img = img.convert('L')  #灰度处理

#(2)图片二值化处理、二维数组转一维数组
import numpy as np
img_new = img.point(lambda x: 0 if x > 128 else 1)  #颜色转换成数字
arr = np.array(img_new) #转换成二维数组
arr_new = arr.reshape(1, -1)  #转换一维数组

#3进行预测
answer = knn.predict(arr_new)  #传入处理好的一维数组
print('图片中的数字为：' + str(answer[0]))  #打印预测结果

#总体来说，K近邻算法是一种非常经典的机器学习算法，其原理清晰简单，容易理解，不过也有一些缺点
#例如，样本量较大时计算量大，拟合速度较慢。