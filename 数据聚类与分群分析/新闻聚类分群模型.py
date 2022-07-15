#1读取数据(数据为文本）
import pandas as pd
df = pd.read_excel('/Users/macbookair/Desktop/数据分析/书本配套资料及电子书/python/Python大数据分析与机器学习/源代码汇总-2020-12-16/第13章 数据聚类与分群/源代码汇总_PyCharm格式/新闻.xlsx')
df.head()
#中文分词
#在前面章节中出现的数据可以分为连续型(如0～100的连续值）和离散型(如样本的类别0和1），但还有一类数据是文本数据。
#而文本类型数据需要转换为数值类型但数据才能在python中处理。这项工作需要用到两个核心技术——中文分词和文本向量化
#首先来讲中文分词：就是将一句话拆分成一些词语，例如'我爱北京天安门'就可以拆分成'我'爱''北京'天安门'。
#中文分词后就可以进行文本向量化，搭建词频矩阵，从而将文本转换为数字

#示例用jieba库进行分词
import jieba
word = jieba.cut('我爱北京天安门')   #用cut()函数对指定的文本内容进行分词
for i in word:
    print(i)


#先对第一条新闻标题进行分词
import jieba
word = jieba.cut(df.iloc[0]['标题'])  #对第1条新闻对标题进行分词，iloc[0]表示选取第一行数据，即第一条新闻对数据['标题']表示选取列名为'标题'
result = ' '.join(word)    #用join函数将变量wrod中的分词以''为连接符连在一起
print(result)

#将上述方法结合for循环遍历整张表格，就能对所有对新闻标题进行分词
wrods = []
for i, row in df.iterrows():  #df.iterrows()是pandas库遍历表格每一行对方法；i对应每一行对行号；row对应每一行的内容
    word = jieba.cut(row['标题'])  #row['标题']表示这一行'标题'的内容
    result = ' '.join(word)       #以空格('')为连接符将变量wrod中的各个分词连接在一起
    wrods.append(result)           #将每一条新闻标题的分词结果添加到wrods列表中
print(wrods[0:3])

#也可以将上述代码合并成三行
wrods = []
for i, row in df.iterrows():
    wrods.append(''.join(jieba.cut(row['标题'])))
print(wrods[0:3])


#补充知识点：遍历DataFrame表格的函数——iterrowe
for i, row in df.iterrows():
    print(i)
    print(row)
#可以看到，这里的i就是每一行的索引号，row就是每一行的内容，该内容是一个一维的Series对象，它可以根据索引来提取内容
#例如，通过row['标题']可以提取该条新闻的标题内容


#文本向量化基础：建立词频矩阵
#此时已经把每一条新闻标题分词完后并存储到words列表中，下面需要将这些文本类型的数据转化为数值类型的数据，
#以便构造特征变量及训练模型。python中有一个文本向量化函数()CountVectorizer（），通过它可以很方便地将文本转换成数值
from sklearn.feature_extraction.text import CountVectorizer   #CountVectorizer引入模块
test = ['金融 科技 厉害', '华能 信托 厉害']      #示例俩条新闻标题，每条新闻标题已经分词完毕
vect = CountVectorizer()                     #函数赋给变量
X = vect.fit_transform(test)                 #fit_transform()函数进行文本向量转换
X = X.toarray()                              #将转换好的x转换为数组
print(X)         #可以看到，新闻标题已经变成量由数字0和1组成的2个一维数组，每个数组中各有5个元素
wrods_bag = vect.vocabulary_       #获取词袋的内容及相应编号
print(wrods_bag)
#这是如何做到的？CountVectorizer()函数会先根据空格来识别每一句话中的词语，例如，它能从第一条新闻标题中识别出'金融'科技'厉害'这3个词
#从第2条新闻标题中识别出'华能'信托'厉害'这三个词，这2条标题便构成了'金融'科技''华能'信托'厉害'这5个不同的词
#而这5个词便构成了这两条新闻标题的词袋，CountVectorizer() 函数会自动对词袋中的词进行编号，通过vocabulary_属性便能获取词袋的内容及相应的编号
#因为CountVectorizer() 函数最开始是设计用来做英文单词向量化的，所以此处的词袋中的词并不是按照拼音首写字母进行排序的
#可以看到词袋是一个字典，每个词是字典的键，词对应的编号是字典的值，这些不同的词其实就代表着不同特征，第几个编号就代表着第几个特征
#     特征1：信托      特征2：华能      特征3：厉害      特征4：科技      特征5：金融
#标题1    0               0               1               1               1
#例如，标题1为'金融'科技'厉害'，特征3、4、5对应的词在标题中各出现一次，上表中在特征处3、4、5对应的值就是1，特征1、2对应的词在标题1中出现0次
#对应的值就是0，所以标题1对应的数值数组就是[0 0 1 1 1]，标题2同理
#此外CountVectorizer()   函数会自动过滤掉一个字的词，这样会过滤掉'的''之'等没有重要含义的词，不过同时也会过滤掉'爱'坑'等可能有重要含义的词
#因此，这个特点既是一个优势，也是一个劣势

#文本向量化实战：构造特征变量
from sklearn.feature_extraction.text import CountVectorizer
vect = CountVectorizer()
X = vect.fit_transform(wrods)   #进行文本向量化处理
X = X.toarray()          #用toarray函数将X转换为数组形式并重新赋给变量X
wrods_bag = vect.vocabulary_   #获取词袋的内容及编号
print(wrods_bag)
print(len(wrods_bag))     #查看词袋中的总词数

#将所有的新闻标题进行文本向量化利用pandas库进行展示
from sklearn.feature_extraction.text import CountVectorizer
vect = CountVectorizer()
X = vect.fit_transform(wrods)    #将分词后的内容文本向量化
X = X.toarray()
wrods_bag2 = vect.get_feature_names_out()   #从词袋中获取不带编号的词
df = pd.DataFrame(X, columns=wrods_bag2)
print(df)
#总结来说，当有n条新闻标题时，先用jieba库对它们进行分词，然后用CountVectorizer()函数提取分词中k个不同的词
#用这些词构成一个词袋，每一个词对应一个编号，即相应的特征，根据原标题中相关词出现的次数来赋值相关特征为i(即相关词出现的次数）
#这样就完成来文本数值化的工作，接下来就可以进行模型的搭建与使用了

#模型的搭建与使用
#用KMeans算法进行聚类分群
from sklearn.cluster import KMeans
kms = KMeans(n_clusters=10, random_state=123)   #n_clusters=10即样本聚合成10类
kms_data = kms.fit_predict(df)  #用fit_predict()函数将模型拟合训练和聚类结果传递合二为一
#等同于先用fit()函数训练模型，再通过labels_属性获取聚类结果的写法（kms.fit(df),label=kms.labels_）
print(kms_data)     #其中每一个数字代表一个分类，可以看到KMeans算法将新闻标题分成类10类

#查看打印分类为1的数据查看该分类中的新闻标题
import numpy as np
wrods_ary = np.array(wrods)    #将生成的存储分词完毕的新闻标题的wrods列表转换成数组
print(wrods_ary[kms_data == 1])  #以逻辑判断的方式筛选分类为1的数据
#可以看到分类为1的新闻标题大多是和python相关的，整体说KMeans算法对新闻标题的聚类效果还是不错的，不过从整体上来看仍有改进的空间
#例如，本案例原始数据中每类新闻的条数基本是一致的，但是聚类分群后很多新闻标题被分类为8，说明有些分类还是不太准确
#可能原因是有的新闻标题文字较多，有的新闻标题文字较少，导致进行距离计算时有些小问题，后面可以通过余弦相似度进行模型优化

#通过DBSCAN算法进行聚类分群
from sklearn.cluster import DBSCAN
dbs = DBSCAN()
dbs_data = dbs.fit_predict(df)   ##用fit_predict()函数将模型拟合训练和聚类结果传递合二为一
print(dbs_data)
#从图中可以看出DBSCAN算法对新闻标题的聚类效果较差，其中有大量离群点(-1)，即不知道这条新闻标题属于什么分类。
#这是因为进行文本向量化后，每个新闻标题都有2015个特征，过多的特征容易导致样本间的距离较远，从而产生离群点。
#因此，对于新闻文本而言KMeans算法的聚类效果很好，DBSCAN算法的聚类效果则不尽人如意，这也说明量对于特征变量较多的数据
#KMeans算法的聚类效果要要优于DBSCAN算法的聚类效果
#不过其还有可以优化的地方，例如，本案例的原始数据集是根据10个关键词爬取的962条新闻，且每类新闻的条数是相近的
#因此最理想的聚类结果是分为10类且每类数据的数量约100个，而上面KMeans聚类结果中很多新闻被分类为7，分类显得不均衡，离这一目标还有些差距

#模型误差产生的原因
#产生这一差距的原因的主要原因是新闻标题长短不一，在进行中文分词及文本向量化后，长标题和短标题的距离较远，而KMeans模型是根据欧式距离进行聚类分析的
#因此容易造成含义相近的长标题和短标题却被分到不同的类别中。用如下所示的3条新闻标题进行演示，其中第3条新闻标题就是把第2条新闻标题重复离两遍
#想去     华能      信托
#华能     信托      很好      想去
#华能     信托      很好      想去      华能     信托      很好      想去
#从含义来说，第2条新闻标题和第三条新闻标题应该是最相似的，但是将它们进行文本向量化后，会发现得到的结果是第二条新闻标题和第一条新闻标题最相似
wrods_test = ['想去     华能      信托', '华能     信托      很好      想去', '华能     信托      很好      想去      华能     信托      很好      想去']
#文本向量化
vect = CountVectorizer()
X_test = vect.fit_transform(wrods_test)  #将分词后的内容文本向量化
X_test = X_test.toarray()       #转换成数组

#查看文本向量化的结果
wrods_bag2 = vect.get_feature_names_out()   #从词袋中获取不带编号的词
df_test = pd.DataFrame(X_test, columns=wrods_bag2)
print(df_test)
#根据打印结果计算新闻标题的欧式距离
#第2条和第一条的距离=√(1-1)^2+(1-1)^2+(1-0)^2+(1-1)^2=1
#第3条和第2条的距离=√(2-1)^2+(2-1)^2+(2-1)^2+(2-1)^2=2
#可以看到第2条新闻标题和第一条新闻标题的距离更近，但其实第2条新闻标题和第3条新闻标题才是最相似的。这种因为文本长短不一造成的预测不精确
#可以通过余弦相似度来解决，余弦相似度是根据向量的夹角来判断相似度的

#补充知识点：用代码计算欧式距离
import numpy as np
dist = np.linalg.norm(df_test.iloc[0] - df_test.iloc[1])
#df_test.iloc[0]表示df_test的第一行数据，df_tes.iloc[1]则表示第二行数据
print(dist)

#余弦相似度代码实现(用cosine_similarity()函数能快速计算各个数据的余弦相似度）其中df_test为上面演示3条新闻标题欧式距离计算是获得的文本向量化结果
from sklearn.metrics.pairwise import cosine_similarity
cosine_similarities = cosine_similarity(df_test)
print(cosine_similarities)
#打印结果为3行3列的二维数组，第i行第j列的数字表示第i个数据和第j个数据的余弦相似度。
#例如，第二行第3列的数字1是第2条新闻标题和第3条新闻标题的余弦相似度
#而第2行第列的数字0.866则是第2条新闻标题和第一条新闻标题的余弦相似度，与数学计算结果一致
#此外，从左上角至右下角线上的数字都为1，这些数字的意义不大，因为它们表示数据与本身的余弦相似度。
#例如第2行第2列的数字表示第2条新闻标题和第2条新闻标题的余弦相似度，必然为1

#利用余弦相似度进行模型优化
from sklearn.metrics.pairwise import cosine_similarity
cosine_similarities = cosine_similarity(df)
print(cosine_similarities)
#用KMeans算法对余弦相似度的计算结果进行聚类分群
from sklearn.cluster import KMeans
kms = KMeans(n_clusters=10, random_state=123)
k_data = kms.fit_predict(cosine_similarities)
print(k_data)


