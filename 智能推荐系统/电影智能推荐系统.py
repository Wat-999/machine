#1读取数据
import pandas as pd
movies = pd.read_excel('/Users/macbookair/Desktop/数据分析/书本配套资料及电子书/python/Python大数据分析与机器学习/源代码汇总-2020-12-16/第14章 智能推荐系统/源代码汇总_PyCharm格式/电影.xlsx')
score = pd.read_excel('/Users/macbookair/Desktop/数据分析/书本配套资料及电子书/python/Python大数据分析与机器学习/源代码汇总-2020-12-16/第14章 智能推荐系统/源代码汇总_PyCharm格式/评分.xlsx')
df = pd.merge(movies, score, on='电影编号')  #merge函数合并表，on参数指定按照哪一列进行合并
print(df.head())
df.to_excel('电影推荐系统.xlsx')     #将汇总表导出一个新的excel表

#在分析数据之前先用数据可视化方法观察数据
import matplotlib.pylab as plt
df['评分'].hist(bins=20)    #用hist()函数绘制直方图，y轴为各评分出现的次数，bins指bin(箱子)的个数，即每张图柱子的个数，figsize指每张图的尺寸大小
plt.show()

#2数据分析
#用pandas库的groupby()函数对合并原始数据得到的Dataframe按照'名称'归类，在用mean()函数求每部电影的评分均值，
#再将得到的新Dataframe赋值给变量ratings
ratings = pd.DataFrame(df.groupby('名称')['评分'].mean())
ratings.sort_values('评分', ascending=False).head()    #ratings.sort_values函数将评分均值从高到低排序，并借助head()函数查看前5行
print(ratings.sort_values('评分', ascending=False).head())

#同样用pandas库的groupby()函数对数据按'名称'归类，在用count()函数统计每部电影对评分次数，然后为每一部电影新增一列'评分次数'
ratings['评分次数'] = df.groupby('名称')['评分'].count()
ratings.sort_values('评分次数', ascending=False).head()
print(ratings.sort_values('评分次数', ascending=False).head())
#从表中可以看出，排除极少数电影评分次数极低对情况，通常某部电影的评分次数越多，该电影的评分也会越高。
#假设某个用户给《阿甘正传》打类高分，我们需要寻找与《阿甘正传》相似度高的电影推荐给该用户

#数据处理
#将原始数据转换为数据透视表。数据透视表是一种交互式表格，我们可以动态调整表格的版面布局，以便通过不同方式分析数据，如求和，计数等
user_movie = df.pivot_table(index='用户编号', columns='名称', values='评分')
#用pandas库中的pivot_table()函数基于变量df创建数据透视表
#index='用户编号'，即以用户编号作为数据透视表的索引
#columns='名称'，即以电影名称作为数据透视表的列
#values='评分'，即以电影评分作为数据透视表中显示的数据
print(user_movie.tail())    #查看数据透视表的最后5行
#如下图所示，其中行代表不同的用户，列代表不同的电影，第i行第j列单元格中的值代表第i个用户对第j部电影对评分
#可以看到绝大部分评分是NaN，数据透视表显得非常稀疏，这是因为电影数量过于庞大，而每个用户打分对电影数量却很有限

#用describe()函数查看该数据透视表对描述性统计信息
user_movie.describe()
print(user_movie.describe())   #std标准差

#系统搭建
#利用之前处理好的数据进行相关性分析，以阿甘正传为例，分析应该向观看列阿甘正传的用户推荐什么电影
#首先从数据透视表中提取各用户对阿甘正传的评分，其中FG是阿甘正传的英文名首字母的缩写
FG = user_movie['阿甘正传（1994）']
pd.DataFrame(FG).head()
print(pd.DataFrame(FG).head())

#用corrwith（）函数计算阿甘正传与其他电影间的皮尔逊相关系数
corr_FG = user_movie.corrwith(FG)    #变量corr_FG是其他电影与阿甘正传的皮儿逊相关系数的series
similarity = pd.DataFrame(corr_FG, columns=['相关系数'])    #整合二维表格
print(similarity.head())
#表中有些相关系数是空值NaN，这是因为计算变量user_movie的列向量和变量FG的皮尔逊相关系数时，
#其实是在计算某部电影的所有评分和阿甘正传的所有评分的皮尔逊相关系数。
#如果某列的空值NaN过多，与阿甘正传的所有用户的评分一个交叉项也没有，即没有一个用户同时对两部电影进行打分，那么就无法计算皮尔逊相关系数
#导致结果中出现列很多NaN值，这些NaN值的数据无效，可以使用dataframe的dorpna（）函数进行剔除
similarity.dropna(inplace=True)   #也可以写成similarity.dropna()  inplace=True表示原地操作不改变表结构
print(similarity.dropna(inplace=True))
#因为行索引是电影名称，可以用merge函数按行索引对齐合并的方式合并表格similarity和ratings，这样就可以把每部电影与阿甘正传的皮尔逊相关系数
#和每部电影评分次数显示在同一张表中
similarity_new = pd.merge(similarity, pd.DataFrame(ratings['评分次数']), left_index=True, right_index=True)
#left_index：使用左则DataFrame中的行索引做为连接键
#right_index：使用右则DataFrame中的行索引做为连接键
#或者用join函数进行连接
#similarity_new = similarity.join(ratings['评分次数'])    #默认是用索引来左连接
print(similarity_new.head())
#因为电影数量庞大，每个用户评过分的电影数量是有限的导致许多电影的评分次数很少，所以可能有偶然的因素导致部分电影的评分偏高偏低
#无法反映真实水平，此时需要设置阀值，只有当该评分大于该阀值时才认为该电影的总体评分有效，这里简单设置阀值为20，然后用sort_values函数将表格按相关关系降序排列
similarity_new[similarity_new['评分次数'] > 20].sort_values(by='相关系数', ascending=False).head()
#sort_values()函数指定列名排序  参数by='相关系数'指定列名   参数ascending=False降序排列
print(similarity_new[similarity_new['评分次数'] > 20].sort_values(by='相关系数', ascending=False).head())
#设置阀值后，与阿甘正传的皮尔逊相关系数较高的前4部电影分别是抓狂双宝、雷神2：黑暗世界、致命吸引力、X战警：逆转未来。
#因此，针对原始数据中9712部电影和100836条评分，使用皮尔逊相关系数作为相似度量的基于物品的协同过滤算法得出的结论是
#阿甘正传与抓狂双宝、雷神2：黑暗世界、致命吸引力、X战警：逆转未来这4部电影的相似度很高，可以认为喜欢阿甘正传的用户较大可能也喜欢这四部电影
#进而可以向对阿甘正传评分很高的用户推荐这四部电影，同样也可以向喜欢这4部电影的用户推荐阿甘正传
#总体来说，我们通常采用协同过滤算法搭建智能推荐系统，并且在大多数应用场景中偏向于使用基于物品的协同过滤算法，也就是寻找不同物品间的相似性
#将相似的物品推荐给用户。

#补充知识点：pandas库的分类函数groupby（）函数
import pandas as pd
data = pd.DataFrame([['战狼2', '丁一', 6, 8], ['攀登者', '王二', 8, 6], ['攀登者', '张三', 10, 8], ['卧虎藏龙', '李四', 8, 8], ['卧虎藏龙', '赵五', 8, 10]], columns=['电影名称', '影评师', '观前评分', '观后评分'])
print(data)

means = data.groupby('电影名称')[['观后评分']].mean()#先按电影名称进行指定分组，再对选取分组后的观后评分求平均值
#注意：最好用两层中括号来选取列，不要写成['观后评分']，否则将获得一个一维的Series对象，而不是一个二维的Dataframe
print(means)

means = data.groupby('电影名称')[['观前评分', '观后评分']].mean()  #先按电影名称进行指定分组，再对选取分组后的多列求平均值
print(means)

means = data.groupby(['电影名称', '影评师'])[['观后评分']].mean()  #先按电影名称、影评师进行指定分组，再对选取分组后的多列求平均值
#其实也就是设置列多重索引，第一重索引为电影名称，第二重索引为影评师
print(means)

count = data.groupby('电影名称')[['观后评分']].count()  #先按电影名称进行指定分组，再对选取分组后的观后评分进行统计每部电影被评分的次数
print(count)

count = count.rename(columns={'观后评分':'评分次数'})  #可以用rename函数修改列名
print(count)



