#1读取数据
import pandas as pd
df = pd.read_excel('/Users/macbookair/Desktop/数据分析/书本配套资料及电子书/python/Python大数据分析与机器学习/源代码汇总-2020-12-16/第4章 逻辑回归模型/源代码汇总_PyCharm格式/股票客户流失.xlsx')

#2划分特征变量和目标变量
x = df.drop(columns='是否流失')  #用drop（）函数删除'是否流失列'，将剩下的数据作为特征变量赋给变量x
y = df['是否流失']        #通过DataFrame提取列的方式提取"是否流失"列作为目标变量，并赋值给y

#3划分训练集和测试集
from sklearn.model_selection import train_test_split   #从sklearn-Learn库引入train_test_split函数
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)   #用train_test_split（）函数划分训练集和测试集

#x_train,y_train为训练集的特征变量和目标变量数据
#x_test,y_test为测试集的特征变量和目标变量数据
#train_test_split(x, y, test_size=0.2, random_state=1)参数中x、y便是之前划分的特征变量和目标变量；test_size则是测试集数据所占的比列，这里选择的是0.2，即20%
#random_state参数赋值为1，该数字没有特殊含义，可以换成其他数字，它相当于一个种子参数，使得每次划分数据的结果一致
#通常会根据样本量的大小来划分训练集和测试集，当样本量较大时，可以划分多一点数据给训练集，
#例如，有10万组数据时，可以按9：1的比列来划分训练集和测试集。这里有7000多组数据，并不算多，所以按照8：2的比例来划分训练集和测试集


#4模型搭建
from sklearn.linear_model import LogisticRegression   #引入逻辑回归模型LogisticRegression
model = LogisticRegression()     #将逻辑模型赋值给变量model，这里没有设参数，即为默认参数
model.fit(x_train, y_train)       #传入训练集的参数x_train, y_train

#5模型使用1：预测数据结果
y_pred = model.predict(x_test)      #传入参数测试集x_test
print('预测数据结果：' + str(y_pred[0:100]))    #查看预测结果的前100项    其中0为预测不会流失，1为预测会流失

#计算全部数据的预测准确度
from sklearn.metrics import accuracy_score    #引入可以计算准确度的accuracy_score
score = accuracy_score(y_pred, y_test)        #将预测值y_pred，和实际值y_test 传入accracy_score()函数，计算预测准确度
print('全部数据的预测准确度:' + str(score))     #打印结果为0.7977，即预测准确度为79.77%，说明近1400组测试数据中，约1117组数据预测正确，283组数据预测错误
#model.score(x_test, y_test)                 #除了用accuracy_score（）函数，还可以使用模型自带的score（）函数，其计算结果是一样的

#6模型使用2：预测概率（逻辑回归模型的本质是预测概率，而不是直接预测具体类别）
y_pred_proba = model.predict_proba(x_test)   #传入测试集特征变量x_test
a = pd.DataFrame(y_pred_proba, columns=['不流失概率', '流失概率'])  #创建二维数组
print(a.head())  #打印结果
#可以看到前5行数据的不流失（分类为0）的概率都大于0.5，因此必然都大于流失（分类为1）的概率，因此这5行数据读会被判定为不流失（分类为0）
#print('预测概率:' + str(y_pred_proba[0:5]))   #打印前5个客户分类的概率

#用ks曲线评估客户流失预警模型关注ks的值   KS=max（TPR-FPR)
#ks曲线和roc曲线在本质上是相同的，同样关注命中率（TPR)和假警报率（FPR),也同样希望命中率tpr尽可能搞，即尽可能揪出潜在流失客户，同时也希望假警报率fpr尽可能低，即不要把未流失客户误判为流失客户
#区别与roc曲线将假警报率fpr作为横坐标，将命中率tpr作为纵坐标，ks曲线将阀值作为横坐标，将命中率tpr与假警报率fpr之差作为纵坐标

from sklearn.metrics import roc_curve   #引入roc_curve ()函数传入测试集的目标变量y_test及预测的流失概率y_pred_proba[:,1]
fpr, tpr, thres = roc_curve(y_test, y_pred_proba[:,1])
#因为roc_curve（）函数返回的是一个含有3个元素的元组，其中默认第一个元素为假警报率，第二个元素为命中率，第三个元素为阀值
#所以这里将三者分别赋给变量fpr（假警报率）、tpr（命中率）、thres（阀值）

#此时获得的fpr、tpr、thres为三个一维数组，将三者合并成一个二维数组
a = pd.DataFrame()   #创建一个空DataFrame
a['阀值'] = list(thres)
a['假警报率'] = list(fpr)
a['命中率'] = list(tpr)
print(a.head())    #打印前5行
print(a.tail())    #打印后5行

#绘制ks曲线
import matplotlib.pyplot as plt
plt.plot(thres[1:], tpr[1:])  #用阀值作横坐标，纵坐标为命中率    表格第一行中的阀值大于1无意义，因此通过切片将第一行剔除
plt.plot(thres[1:], fpr[1:])  #用阀值作横坐标，纵坐标为假警报率
plt.plot(thres[1:], tpr[1:]-fpr[1:])    #用阀值作横坐标，纵坐标为命中率与假警报率之差
plt.title('KS')    #添加标题
plt.xlabel('threshold')   #添加x轴标签
plt.legend(['tpr', 'fpr', 'tpr-fpr'])  #添加图列
plt.gca().invert_xaxis()  #反转x轴：先用gca（）函数（gca代表get current axes）获取坐标轴信息，再用invert_xaxis反转x轴
plt.show()

#快速求ks值
print('ks值：' + str(max(tpr - fpr)))   #打印ks值，结果为0.4744，在[0.3,0.5]区间内，因此该模型具有较强的分区能力
#一般情况下，我们希望模型有较大的KS值，因为较大的KS值说明模型有较强的分区能力，不同取值范围的ks值的含义如下：
#ks值小于0.2,一般认为模型的区分能力较弱
#ks值在[0.2,0.3]区间内，模型具有一定区分能力；
#ks值在[0.3,0.5]区间内，模型具有较强的区分能力；
#但ks值也不是越大越好，如果ks值大于0.75,往往表示模型有异常。在商业实战中，ks值处于[0.2,0.3]区间内已经算是挺不错的


#补充知识点：获取ks值对应的阀值
a['TPR-FPR'] = a['命中率'] -a['假警报率']
print(a.head())
#print(max(a['TPR-FPR']))    #打印表格后，直接打印也可以得到和print('ks值：' + str(max(tpr - fpr))) 一样的结果


