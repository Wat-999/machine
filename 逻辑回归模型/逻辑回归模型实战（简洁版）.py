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