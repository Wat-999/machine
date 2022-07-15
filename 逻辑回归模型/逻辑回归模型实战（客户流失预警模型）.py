#1数据读取与变量划分
#1读取数据
import pandas as pd
df = pd.read_excel('/Users/macbookair/Desktop/数据分析/书本配套资料及电子书/python/Python大数据分析与机器学习/源代码汇总-2020-12-16/第4章 逻辑回归模型/源代码汇总_PyCharm格式/股票客户流失.xlsx')

#2划分特征变量和目标变量
x = df.drop(columns='是否流失')   #用drop（）函数删除'是否流失列'，将剩下的数据作为特征变量赋给变量x
y = df['是否流失']    #通过DataFrame提取列的方式提取"是否流失"列作为目标变量，并赋值给y

#2模型的搭建与使用
#1划分训练集和测试集合
from sklearn.model_selection import train_test_split  #从sklearn-Learn库引入train_test_split函数
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)  #用train_test_split（）函数划分训练集和测试集
#x_train,y_train为训练集的特征变量和目标变量数据
#x_test,y_test为测试集的特征变量和目标变量数据
#train_test_split(x, y, test_size=0.2, random_state=1)参数中x、y便是之前划分的特征变量和目标变量；test_size则是测试集数据所占的比列，这里选择的是0.2，即20%
#random_state参数赋值为1，该数字没有特殊含义，可以换成其他数字，它相当于一个种子参数，使得每次划分数据的结果一致
#通常会根据样本量的大小来划分训练集和测试集，当样本量较大时，可以划分多一点数据给训练集，
#例如，有10万组数据时，可以按9：1的比列来划分训练集和测试集。这里有7000多组数据，并不算多，所以按照8：2的比例来划分训练集和测试集
x_train.head()
y_train.head()
x_test.head()
y_test.head()
print(x_train.head(), y_train.head(), x_test.head(), y_test.head())  #打印划分后的训练集和测试集

#划分训练集和测试集在某种程度上也是为了检查模型是否出现过拟合。过拟合指模型在训练样本中拟合程度过高，虽然它很好地契合了训练集数据
#但是却丧失了泛化能力，因而不具有推广性，导致在测试集数据中的预测表现不佳。就好比每一次模考都做同一份卷子，训练时得分很高，
#但是期末考试换了一套卷子就得分很低。而划分训练集和测试集可以用来对模型进行更好对检验

#2模型搭建
from sklearn.linear_model import LogisticRegression  #引入逻辑回归模型LogisticRegression
model = LogisticRegression()   #将逻辑模型赋值给变量model，这里没有设参数，即为默认参数
model.fit(x_train, y_train)    #传入训练集的参数x_train, y_train

#3模型预测
y_pred = model.predict(x_test)  #传入参数测试集x_test
print(y_pred)
#print(y_pred[0:100])      #查看预测结果的前100项    其中0为预测不会流失，1为预测会流失

#4将模型的预测值y_pred和测试集的实际值y_test进行汇总。
a = pd.DataFrame()   #创建一个空DataFrame 把a['预测值']，a['实际值']集成到DataFrame中
a['预测值'] = list(y_pred)    #传入测试集的预测值 y_pred是一个numpy.ndarray类型的一维数组结构
a['实际值'] = list(y_test)    #传入测试集的实际值 y_test为Series类型的一维序列结构，用list（）函数将它们都转换为列表
print(a.head())       #查看表格的前5行   前5项的预测准确度为80%

from sklearn.metrics import accuracy_score  #引入可以计算准确度的accuracy_score
score = accuracy_score(y_pred, y_test)      #将预测值y_pred，和实际值y_test 传入accracy_score()函数，计算预测准确度
#model.score(x_test, y_test)         #除了用accuracy_score（）函数，还可以使用模型自带的score（）函数，其计算结果是一样的
print(score)  #打印结果为0.7977，即预测准确度为79.77%，说明近1400组测试数据中，约1117组数据预测正确，283组数据预测错误

#5预测概率（逻辑回归模型的本质是预测概率，而不是直接预测具体类别）
y_pred_proba = model.predict_proba(x_test)     #传入测试集特征变量x_test
a = pd.DataFrame(y_pred_proba, columns=['不流失概率', '流失概率'])  #创建二维数组
print(a.head())  #打印结果
#可以看到前5行数据的不流失（分类为0）的概率都大于0.5，因此必然都大于流失（分类为1）的概率，因此这5行数据读会被判定为不流失（分类为0）

#只查看流失（分类为1）概率，可以采用如下代码
#y_pred_proba[:, 1]  #[:, 1]中的'：'表示二维数组所有的行，'1'表示二维数组的第二列（即流失概率的列）

#6获取逻辑回归系数
print(model.coef_)        #系数k1   model为之前训练的模型名称
print(model.intercept_)   #截距项k0

#批量查看预测概率
import numpy as np
for i in range(5):   #计算前5条测试集数据的预测概率作为演示
    print(1/(1+np.exp(-(np.dot(x_test.iloc[i], model.coef_.T) + model.intercept_))))  #预测 y=1（流失）的概率
#np.exp（）用于进行指数运算（即e^x)
#x_test.iloc[i] ：x_test传入测试集参数，  iloc[i]用来选取DataFrame的行数据
#np.dot()用于进行数据点乘运算，即将系数和特征值一一相乘
#model.coef_.T中.T则是将数据进行转置，为点乘运算做准备


#补充知识点：roc曲线的混淆矩阵
from sklearn.metrics import confusion_matrix
m = confusion_matrix(y_test, y_pred)  #传入预测值和实际值
a = pd.DataFrame(m, index=['0（实际不流失）', '1(实际流失）'], columns=['0(预测不流失）', '1(预测流失）'])
print(a)
#可以看到，实际流失348（192+56）人中有156人被准确预测，命中率（TPR)为45%；
#实际不流失的1061（968+93）人中有93人被误判为流失，假警报率（FPR)为8.8%。
#需要注意的是，这里的TRR和FRP都是在阀值为50%的条件下计算的

#也可以通过如下代码计算
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))   #传入预测值和实际值

#运行结果如下图所示，其中recall对应的就是命中率（又称召回率），可以看到对于分类为1的命中率仅为0.45,和之前手动计算一致
#accuracy表示整体准确度，其值为0.8，和上面计算的4模型计算的预测度0.7977是一致的；
# support表示样本数，其中1061为实际分类为0的样本，348为实际分类为1的样本数，1409为测试集的全部样本数。
#性能度量指标-precision（精准率）和fl——score，这俩个指标相对于命中率（TPR)和假警报率（FPR)的重要性低很多，简单即可
#precision（精准率），公式TP/(TP+FP),含义预测为流失（1）中实际流失（1）的比例
#fl-score，公式2TP/(2TP+FP=FN),含义混合的度量，对不平衡类别比较有效

