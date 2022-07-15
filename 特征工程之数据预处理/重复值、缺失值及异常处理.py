#在数据录入和 处理过程中，不可避免地会产生重复值、缺失值、及异常值

#重复值处理
import pandas as pd
df = pd.DataFrame([[1, 2, 3], [1, 2, 3],[4, 5, 6]], columns=['c1', 'c2', 'c3'])
print(df)

#如果数据量较大时，可以用duplicated（）函数来查重复的内容
#df = df[df.duplicated()]      #将重复的第二行筛选出来了
#print(df)
#若要统计重复行的数量，用sum函数
#df = df.duplicated().sum()
#print(df)

#发现有重复行时，用drop_duplicates()函数删除重复行
df = df.drop_duplicates()  #注意：drop_duplicates()函数并不改变原表格结构，所以需要重新赋值，或者在其中设置inplace参数为True
#df = df.drop_duplicates('c1')    #按列去重
print(df)

#缺失值处理
import numpy as np
data = pd.DataFrame([[1, np.nan, 3], [np.nan, 2, np.nan], [1, np.nan, 0]], columns=['c1', 'c2', 'c3'])
#np.nan代表缺失值，又叫空值
print(data)

#用isnull（）或isna（）函数（两者作用类似）来查看空值
print(data.isnull())    #isnull的作用是判断是否空值，若是空值就赋予True，否则就赋予False
#对单列查看空值
print(data['c1'].isnull())

#如果数据量较大，可以通过如下代码筛选出某列中内容为空值对行
print(data[data['c1'].isnull()])
#其实本质上是根据data['c1'].isnull()得到的True和False来筛选，如果是True则被筛选出来，输出结果可以看到只要是
#c1列是空值的行都被筛选出来了
#对于空值有两种常见的处理方式，删除空值和填补空值
#a = data.dropna()    #删除空值  这种删除方法是只要含有空值的行都会被删除，运行结果可以看到，因为每行都有空值，所以都被删除了
#print(a)
#如果觉得上述方法过于激进，可以设置thresh参数，例如将其设置为n，表示如果一行中的非空值少于n个则删除该行
#a = data.dropna(thresh=2)
#如果一行中的非空值少于2个则删除该行，。构造的演示数据中，第1行和第3行都有2个非空值，因此不会被删除，而第二行只有一个空值，少于2个，因此会被删除

#用fillna()函数可以填补空值。这里采用的是均值填补法，用每列的均值对该列的空值进行替换，
#也可以把其中的data.mean()换成data.median(),变成中位数填补
#b = data.fillna(data.mean())     #均值填补法
#c = data.fillna(data.mean())     #中位数填补法
#d = data.fillna(method='pad')    #用空值上方的值来替换空值，如果上方的值不存在或也为空值，则不替换
#e = data.fillna(method='backfill')  #表示用空值下方的值来替换空值，如果下方的值不存在或也为空值，则不替换
#f = data.fillna(method='bfill')     #表示用空值下方的值来替换空值，如果下方的值不存在或也为空值，则不替换，与上述结果一样

#异常值处理
import pandas as pd
df = pd.DataFrame({'c1': [3, 10, 5, 7, 1, 9, 69], 'c2': [15, 16, 14, 100, 19, 11, 8],
'c3': [20, 15, 18, 21, 120, 27, 29]}, columns=['c1', 'c2', 'c3'])
print(df)

#可以看到，第1列的数字69、第2列的数字100、第3列的数字120为比较明显的异常值

#检测异常值
#1利用箱体图观察
#import matplotlib.pylab as plt
#df.boxplot()
#plt.show()

#2利用标准差检测
#当数据服从正态分布时，99%的数值与均值的距离应该在3个标准差之内，95%的数值与均值的距离应该在2个标准差之内，
#因为3个标准差过于严格，此处将阀值设定为2个标准差，即认为当数值与均值的距离超出2个标准差，则可以认为它是异常值
a = pd.DataFrame()
for i in df.columns:        #通过for循环依次对数据的每列进行操作
    z = (df[i] - data[i].mean()) / df[i].std()    #用mean()函数(获取均值)和std()函数(获取标准差)将每列数据进行Z-score:X*=(x-mean)/std
    a[i] = abs(z) > 2 #进行逻辑判断，如果Z—score标准化后的数值大于标准正态分布的标准差1的2倍，那么该数值为异常值，返回布尔值True，否则返回布尔值False
    print(a)   #输出结果可以看到每列各有一个异常值，且被标识为布尔值True
#检测到异常值后，如果异常值较少或影响不大，也可以不处理。如果需要处理可以采用如下几种常见的方式
#删除含有异常值的记录
#将异常值视为缺失值，按照缺失值处理
#按照数据分箱方法进行处理

