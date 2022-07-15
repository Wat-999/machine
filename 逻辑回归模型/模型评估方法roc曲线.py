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
print(y_pred)
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

#7用roc曲线评估客户流失预警模型关注AUC的值   （求不同阀值下的命中率（TPR)和假警报率（FPR)的值从而绘制出roc的曲线)
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
#可以看到，随着阀值的下降，命中率和假警报率都在上升。

#表格解说
#第一行，表格第一行的阀值表示只有当一个客户被预测流失的概率>=193%，才会被判定其会流失，但因为概率不会超过100%，所以此时所有客户都不会被预测流失，
#此时命中率和假警报率都为0，可见这个阀值其实没什么意义，那么为什么还要设置它率？这个阀值是roc_curve ()函数默认设置，官方也是介绍第一个阀值是没有意义的
#第二行数据表示只有当一个客户被预测流失的概率>=93.03%，才会被判定其会流失，这个条件还是比较苛刻的，
#此时被预测为流失的客户还是很少，命中率为0.0028，即'预测为流失且实际流失的客户/实际流失的客户'这一比率为0.0028，
#假设此时共有5000组实际流失的客户，那么在该阀值条件下，实际流失的客户中会有5000*0.0028=14人被准确判定为会流失。此时假警报率为0，
# 即'预测为流失但实际未流失但客户/实际未流失但客户'这一比率为0，即实际未流失的客户中不会有人被误判为流失。表格后续内容的含义可以参考第2行的内容的含义来理解

#绘制roc曲线
import matplotlib.pyplot as plt
plt.plot(fpr, tpr)  #用plot（）函数绘制折线图
plt.title('ROC')    #添加标题
plt.xlabel('FPR')   #添加x轴标签
plt.ylabel('TPR')   #添加y轴标签
plt.show()

#计算模型ACU值
from sklearn.metrics import roc_auc_score
score = roc_auc_score(y_test,y_pred_proba[:,1])  #引入roc_auc_score ()函数传入测试集的目标变量y_test及预测的流失概率y_pred_proba[:,1]
print(score)  #获得的AUC的值为0.81，可以说预测效果还是不错的


#补充知识点阀值点取值方法
#上面已经知道假警报率和命中率是根据阀值计算而来的，那么阀值是如何取之的呢？
#下面来来解释#7用roc曲线评估客户流失预警模型（求不同阀值下的命中率（TPR)和假警报率（FPR)的值从而绘制出roc的曲线）
a = pd.DataFrame(y_pred_proba, columns=['分类为0的概率', '分类为1的概率'])   #为设置列名
a = a.sort_values('分类为1的概率', ascending=False)  #sort_values（）函数指定列名，并设置ascending参数为False进行降序排列
print(a.head(15))    #查看前15行

#下表的第一列为测试集样本的序号，后2列分别为分类为0和分裂为1的概率，可以看到序号326的样本其分类为1的概率最高，为0.9930369,这个概率就是之前提到的阀值（在之前的表格中，它是除1.930369以外最大的阀值）
#所有样本的分类就是根据这个阀值进行的，分类为1的概率小于该阀值的样本都被列为分类0，大于等于该阀值的样本都被列为分类1，因为只有序号326的样本满足分类为1的概率大于等于该阀值（实际上该样本也的确为分类1）其余样本都被列为分类0
#事实上一共有348个实际分类为1的样本，所以此时命中率（TPR)为1/348=0.002874,与程序获得的一致，
#至此可以得出结论，这些阀值都是各个样本分类为1的概率（其实并没有全部提取，例如，序号366的样本分类为1的概率就没有被取为阀值）


