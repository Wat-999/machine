#1读取数据
import pandas as pd
data = pd.read_excel('/Users/macbookair/Desktop/数据分析/书本配套资料及电子书/python/Python大数据分析与机器学习/源代码汇总-2020-12-16/第13章 数据聚类与分群/源代码汇总_PyCharm格式/客户信息.xlsx')

#2绘制散点图
import matplotlib.pylab as plt
data.head()
plt.scatter(data.iloc[:, 0], data.iloc[:, 1], c='green', marker='*')  #iloc[:, 0]用来选取所有行和第1列数，iloc[:, 1]同理
plt.xlabel('age')     #添加x轴标签
plt.ylabel('salary')    #添加y轴标签
plt.show()

#3模型搭建与使用
from sklearn.cluster import KMeans
kms = KMeans(n_clusters=3, random_state=123)   #设置样本分类数，和随机参数保证每次运行得到的聚类结果一致
kms.fit(data)
label = kms.labels_    #获取聚类效果
print(label)

#4用散点图展示KMeans算法的聚类结果
plt.scatter(data[label == 0].iloc[:, 0], data[label == 0].iloc[:, 1], c='red', marker='o', label='class0')
plt.scatter(data[label == 1].iloc[:, 0], data[label == 1].iloc[:, 1], c='green', marker='*', label='class1')
plt.scatter(data[label == 2].iloc[:, 0], data[label == 2].iloc[:, 1], c='blue', marker='+', label='class2')
plt.xlabel('age')     #添加x轴标签
plt.ylabel('salary')    #添加y轴标签
plt.legend(loc=4)      #将图例设置在散点图的右下角
plt.show()

#计算聚类结果的均值
print(data[label == 0].iloc[:, 1].mean())   #data[label == 0表示筛选分类为0的客户，iloc[:, 1]表示选取data表格的所有行和第2列，mean函数用于求平均值
print(data[label == 1].iloc[:, 1].mean())
print(data[label == 2].iloc[:, 1].mean())
#结果解读：class1代表的这部分客户年龄为40～50岁，平均收入在58万元，可以视为重点客户，是需要重点营销和推广的对象
#class2代表的这部分客户年龄为25～42岁，平均收入46万元，可以视为优质客户，是需要精心维护和营销的对象
#class0代表的这部分客户年龄为20～40岁，平均收入21万元，可以视为潜力客户，是需要耐心挖掘和等待的现象