#案例背景
#用户在电商平台发布的产品评价和评分中包含着用户的偏好信息，利用情感分析模型可以从产品评价和评分中获取用户的情感及对产品属性的偏好
#在此基础上，就可以进一步利用智能推荐系统向用户推荐更多他们喜欢的产品，以增加用户的黏性，挖掘潜在利润。

#1读取数据
import pandas as pd
df = pd.read_excel('/Users/macbookair/Desktop/数据分析/书本配套资料及电子书/python/Python大数据分析与机器学习/源代码汇总-2020-12-16/第16章 神经网络模型/源代码汇总_PyCharm格式/产品评价.xlsx')
df.head()   #1代表好评，0代表差评

#2中文分词(文本数据不能直接拿来训练，需要将文本分词，构建词频矩阵，再用来拟合模型）
#import jieba
#word = jieba.cut(df.iloc[0]['评论'])  #用cut函数进行分词，df.iloc[0]['评论']表示选取第一条评论
#result = ','.join(word)    #将word中的各个分词通过空格('')连接在一起
#print(result)

#可以看到已经将第一条评论分词完毕，下面通过for循环遍历整张表格
#words = []
#for i, row in df.iterrows():    #df.iterrows()是pandas库遍历表格每一行的方法，i是每一行的行号，row是每一行的内容
#    word = jieba.cut(row['评论'])
#    result = ''.join(word)
#    words.append(result)
#    print(words)

#也可以简写如下
import jieba
words = []
for i, row in df.iterrows():
    words.append(' '.join(jieba.cut(row['评论'])))

#构造特征变量和目标变量
from sklearn.feature_extraction.text import CountVectorizer
vect = CountVectorizer()
X = vect.fit_transform(words)    #进行文本向量化处理
X = X.toarray()        #将向量化后的文本转换为数组形式
print(X)

words_bag = vect.vocabulary_  #vocabulary_属性便能获取词袋的内容及相应编号
print(words_bag)
print(len(words_bag))       #查看词袋中一共有多少个词
#想更好地查看X，就将其装换成Dataframe格式
#import pandas as pd
#pd.DataFrame(X)  #打印输出，就是所有评论的词频矩阵，也是之后用来搭建模型的特征变量数据集，可以看到第二行中4068列的值为1，即这个词在第2条评论中出现列1次
#print(pd.DataFrame(X).head())

#查看全部列或行，如果将其中的None改成500，则表示最多显示500行或500列
#pd.set_option('display.max_columns', None)   #显示所有列
#pd.set_option('display.max_rows', None)      #显示所有行
#pd.DataFrame(X)

#2目标变量提取
#与特征变量的提取相比，目标变量的提取则容易得多，只需要将最开始读取的表格中"评价"列提取出来即可
y = df['评价']

#神经网络模型的搭建与使用
#1划分训练集和测试集
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=123)

#2搭建神经网络模型
from sklearn.neural_network import MLPClassifier  #引入MLP神经网络模型
mlp = MLPClassifier(random_state=123)
mlp.fit(X_train, y_train)
#参数名称activation：该参数用于指定激活函数，默认值为'relu'也就是relu函数，除此之外，该参数还可以设置为'identity'(不使用激活函数）
#'logistic'(使用Sigmoid激活函数）、'tanh'(使用Tanh激活函数）
#hidden_layer_sizes:该参数用于指定隐藏层的节点数和层数，默认值100，即模型只有1个掩藏层，且隐藏层中的节点数为100。如果将其设置为(100,100,100)
#则表示模型有3个隐藏层，每层有100个节点
#alpha：该参数是正则化参数，默认值为0.0001。提高该参数的值可以防止过拟合，但是会降低该模型的预测准确度
#本案列使用参数的默认值就能获得不错的效果。如果想进一步优化即用多参数调优

#3模型使用
#对测试集数据进行预测
y_pred = mlp.predict(X_test)
print(y_pred)
#汇总预测值和实际值
a = pd.DataFrame()
a['预测值'] = list(y_pred)
a['实际值'] = list(y_test)
print(a.head())

#查看所有测试集数据的预测准确度
from sklearn.metrics import accuracy_score
score = accuracy_score(y_pred, y_test)  #也可以用mlp.score(X_test, y_test)
print(mlp.score(X_test, y_test))
print(score)

#还可以输入一些数据集以外的评价，看看模型能否给出准确的判断
comment = input('请输入您对本商品的评价：')
comment = [' '.join(jieba.cut(comment))]   #分词处理
print(comment)
X_try = vect.transform(comment)        #将分词结果转换为词频矩阵
y_pred = mlp.predict(X_try.toarray())  #将词频矩阵转换为数组再传入模型进行预测
print(y_pred)


#模型对比（朴素贝叶斯模型）
from sklearn.naive_bayes import GaussianNB
nb_clf = GaussianNB()
nb_clf.fit(X_train, y_train)

y_pred = nb_clf.predict(X_test)
print(y_pred)

from sklearn.metrics import accuracy_score
score = accuracy_score(y_pred, y_test)
print(score)   #为0.87说明高斯朴素贝叶斯模型到预测效果略逊于MLP神经网络模型

#总体来说，神经网络模型是一种非常不错到机器学习模型，其学习速度快，预测效果好，不过与其他传统到机器学习模型相比
#其可解释性不高，因而也常被称为"黑盒模型"。



