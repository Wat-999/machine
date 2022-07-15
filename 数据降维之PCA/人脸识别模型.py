#搭建的人脸识别模型是一个比较复杂的n维数据降维到k维数据的案例。
#先对人脸数据进行读取和处理，再通过PCA进行数据降维，最后用K近邻算法搭建模型人脸识别
#人脸识别在本质上是根据每张人脸图像中不同像素点的颜色进行数据建模与判断。人脸图像的每个像素点的颜色都有不同的值
#这些值可以组成人脸的特征向量，不过因为人脸图像的像素点很多，所以特征变量也很多，需要利用PCA进行数据降维。
#与利用K近邻算法识别手写数字类似，首先需要将图片类型的数据转换成数值类型的数据，这样才能方便进行之后的数据降维及模型搭建。

#1读取人脸照片数据
#方法一
import os
names = os.listdir('/Users/macbookair/Desktop/数据分析/书本配套资料及电子书/python/Python大数据分析与机器学习/源代码汇总-2020-12-16/第12章 数据降维之PCA主成分分析/源代码汇总_PyCharm格式/olivettifaces')
print(names[0:5])
#print(os.getcwd())
from PIL import Image
img0 = Image.open(r"/Users/macbookair/Desktop/数据分析/书本配套资料及电子书/python/Python大数据分析与机器学习/源代码汇总-2020-12-16/第12章 数据降维之PCA主成分分析/源代码汇总_PyCharm格式/olivettifaces/" + names[0])
#注意文件目录的路径写法：前面加r，在目录后加/   读取目录下一张图片
img0.show()   #show()函数显示图片
print(img0)

#方法二(等同上面）
#from pathlib import Path
#from PIL import Image
#import os
#names = list(Path('/Users/macbookair/Desktop/数据分析/书本配套资料及电子书/python/Python大数据分析与机器学习/源代码汇总-2020-12-16/第12章 数据降维之PCA主成分分析/源代码汇总_PyCharm格式/olivettifaces').glob('*.*'))
#img0 = Image.open(names[0].as_posix())
#img0.show()
#print(img0)


#2人脸数据处理：特征变量提取
import numpy as np
img0 = img0.convert('L')
#对读取对图片进行灰度转换，参数'L'指转换成灰度格式的图像。灰度处理后，每个像素点的颜色就可以用0～255的数值表示，其中0代表黑色，255代表白色，就完成了图像转换成数字的第一步
img0 = img0.resize((32, 32))  #调整图像尺寸为32*32像素
arr = np.array(img0)         #将1024个像素点的灰度值转换为一个二维数组
print(arr)                   #打印结果中，每个数值都是图像中每个像素点的灰度值

#觉得numpy格式的arr不好观察，就转换成dataframe
import pandas as pd
print(pd.DataFrame(arr))  #此时二维表格共有32行32列，每个单元格中的数值就是该像素点的灰度值

#上面获得的32*32的二维数组还需要转换成1*1024格式才能用于数据建模
arr = arr.reshape(1, -1)

#因为总共有400张图片的灰度值需要处理，若将400个二维数组堆叠起来会形成三维数组，所以我们需要用flatten函数将1*1024的二维数组降维成一维数组
#并用tolist函数将其转换成列表
print(arr.flatten().tolist())
#这样就完成列第一张图片的图像数据到数值类型数据的装换，为方便理解，其实就是第一张图片共有1024个特征变量，每个变量为不同像素点的灰度值

#再将上述方法结合for循环，就可以将所有人脸图片的图像数据都转换成数值类型数据，从而构造相应的特征变量
X = []
for i in names:
    img = Image.open(r"/Users/macbookair/Desktop/数据分析/书本配套资料及电子书/python/Python大数据分析与机器学习/源代码汇总-2020-12-16/第12章 数据降维之PCA主成分分析/源代码汇总_PyCharm格式/olivettifaces/" + i)
    img = img.convert('L')    #灰度处理
    img = img.resize((32, 32))    #设置像素
    arr = np.array(img)          #装换成二维数组
    X.append(arr.reshape(1, -1).flatten().tolist())    #先转换成1*1024格式，再将其由二维数组降维至一维数组，再将其转换成列表

X = pd.DataFrame(X)   #将其转化成DataFrame格式再查看， 比array格式好看
print(X)
print(X.shape)   #查看行列数


#3人脸数据处理：目标变量提取
#提取完特征变量X后，还需要提取目标变量Y（代表人脸所对应的人的编号），他的提取相对容易多
#首先来提取第一张人脸图拍呢的目标变量。该图片的文件名为10_6.jpg，其中10是该图片对应的人的编号，即我们所需要的目标变量
#因为编号与之后的内容都会以'_'号隔开，所以可以用split函数根据'_'号进行字符串分割，从而提取需要的编号
names[0].split('_')[0]
#names[0]为第一张图片的文件名10_6.jpg，split函数根据'_'号将文件名分割为2个部分，通过[0]提取第一部分，即人的编号10
#但是目标变量y需要为数字，所以还需要用int函数将字符串转换为数字
int(names[0].split('_')[0])

#将上述方法结合for循环，便能提取400张人脸图片的目标变量
y = []
for i in names:
    img = Image.open(r"/Users/macbookair/Desktop/数据分析/书本配套资料及电子书/python/Python大数据分析与机器学习/源代码汇总-2020-12-16/第12章 数据降维之PCA主成分分析/源代码汇总_PyCharm格式/olivettifaces/" + names[0])
    y.append(int(i.split('_')[0]))
    print(y)


#数据划分与降维
#获取到特征变量X和目标变量y后，就可以通过常规的机器学习手段来搭建机器学习模型。在正式搭建模型之前，先通过PCA进行数据降维

#1划分训练集和测试集
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

#2PCA数据降维
#X共有1024列，即有1024个特征变量，这么多特征变量可能会带来过拟合即提高模型的复杂度问题，因此需要对特征变量进行PCA降维
from sklearn.decomposition import PCA
pca = PCA(n_components=100)    #即将1024个特征进行线性组合，生成互不相关的100个新特征
pca.fit(X_train)              #使用训练集的特征数据来拟合pca模型

#用拟合好的pca模型分别对训练集和测试集的特征数据进行降维
X_train_pca = pca.transform(X_train)
X_test_pca = pca.transform(X_test)
#验证是否降维成功
print(X_train_pca.shape)
print(X_test_pca.shape)

#模型的搭建与使用
#将训练集和测试集的特征数据降维后可以使用k近邻算法分类模型的搭建。k近邻算法分类模型通过训练掌握某张人脸的部分特征数据
#在面对测试集中的特征数据时就可以根据近邻的思想进行人脸分类

#1模型搭建
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier()
knn.fit(X_train_pca, y_train)  #传入降维后的训练集数据，和目标变量的训练集数据

#2模型预测
y_pred = knn.predict(X_test_pca)    #传入降维后的测试集数据
print(y_pred)
#汇总预测值和实际值
a = pd.DataFrame()
a['预测值'] = list(y_pred)
a['实际值'] = list(y_test)
print(a.head())

#查看所有测试集数据的预测准确度
from sklearn.metrics import accuracy_score
score = accuracy_score(y_pred, y_test)
print('预测准确度：' + str(score))
