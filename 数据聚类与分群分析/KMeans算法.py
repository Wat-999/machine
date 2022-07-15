#机器学习分为监督式学习和非监督式学习两大类。其中非监督式学习的数据集只有特征变量，而没有目标变量，我们需要对已有数据进行建模，
#根据性质进行分组。其典型案例就是聚类分析问题，例如，根据信用卡申请人信息对申请人进行分类（业内常称为客户分群），根据新闻标题和内容对新闻进行分类
#KMeans算法是最常用对一种聚类算法，其中K代表类别数量，Means代表每个类别内样本的均值，所以KMeans算法又称为K-均值算法。
#KMeans算法以距离作为样本间相似度的度量标准，将距离相近的样本分配至同一个类别。样本间距离的计算方式可以是欧式距离、曼哈顿距离、余弦相似度等，
#KMeans算法通常采用欧式距离来度量各样本间的距离
#KMeans算法的核心思想是对每一个样本点计算到各个中心的距离，并将该样本点分配给距离最近的中心点代表的类别，一次迭代完成后，根据聚类结果更新每个类别的中心点
#然后重复之前操作再次迭代，直到前后两次分类结果没有差别。

import numpy as np
data = np.array([[3, 2], [4, 1], [3, 6], [4, 7], [3, 9], [6, 8], [6, 6], [7, 7]])
print(data)

#绘制散点图
import matplotlib.pylab as plt
plt.scatter(data[:, 0], data[:, 1], c='red', marker='o', label='samples')   #以红色圆圈样式绘制散点图并加上标签
#因为data是numpy库构造的，所以data[0:, 1]表示两列数的第1列数（第一个元素表示行，冒号表示所有行；第二个元素表示列，0表示第1列），同理data[:,1]表示y坐标
#设置参数marker表示数据点的形状，label则表示数据标签
plt.legend()   #设置图例
plt.show()

#开始调用KMeans算法的聚类运算
from sklearn.cluster import KMeans
kms = KMeans(n_clusters=2)     #参数n_clusters=2，也就是选取的K值，即将样本分成2类，如果不设置默认为8
kms.fit(data)    #进行模型训练
label = kms.labels_    #通过模型labels_属性获取聚类结果，并赋值给label
print(label)    #结果解读，前两个数值为1，其他数值为0，代表原始数据中前2个数据聚为一类，其他数据聚为另一类

#使用散点图展示KMeans算法聚类的效果
plt.scatter(data[label == 0][:, 0], data[label == 0][:, 1], c='red', marker='o', label='class0')
#data[label == 0]是为来提取被KMeans算法分类为0的原始数据，同理data[label == 1]是为了提取被KMeans算法分类为1的原始数据
#注意这里使用的是双等号'=='，即逻辑判断是否相等
plt.scatter(data[label == 1][:, 0], data[label == 1][:, 1], c='green', marker='*', label='class1')
plt.legend()    #设置图例
plt.show()

#下面将k值设置为3,即将原始数据分为3类
kms_3 = KMeans(n_clusters=3)
kms_3.fit(data)
label_3 = kms_3.labels_   #获取聚类结果
print(label_3)

#仍用散点图展示
plt.scatter(data[label_3 == 0][:, 0], data[label_3 == 0][:, 1], c='red', marker='o', label='class0')
plt.scatter(data[label_3 == 1][:, 0], data[label_3 == 1][:, 1], c='green', marker='*', label='class1')
plt.scatter(data[label_3 == 2][:, 0], data[label_3 == 2][:, 1], c='blue', marker='+', label='class2')
plt.legend()   #设置图例
plt.show()

#注意：KMeans算法的初始中心点是随机的，所以如果样本数据量较大，可能会导致每次运行代码得到的聚类结果略有不同，
#如果希望每次得到的聚类结果都是一样的，可以在模型中传入参数random_state=123