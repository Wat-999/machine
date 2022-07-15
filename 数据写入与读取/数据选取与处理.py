#创建Dataframe
import pandas as pd
data = pd.DataFrame([[1, 2, 3], [4, 5, 6], [7, 8, 9]], index=['r1','r2', 'r3'], columns=['c1', 'c2', 'c3'])
# print(data)

"数据选取"
#a = data['c1']   #按列选取数据
# print(a)

#c = data[['c1', 'c3']]    #若要选取多列，需要在中括号【】中指定列表，选取多列
# print(c)

#b = data.iloc[1:3]     #按行选取数据，根据行序号来选取数据，注意序号从0开始，左闭右开  pandas库推荐用iloc方法来根据行序号选取数据
# print(b)

#d = data.iloc[['r2', 'r3']]    #按行名称来选取数据
# print(d)

#e = data.head()        #如果行数很多，可以通过head函数选取，（）即选取所有，（2）即选取2行
# print(e)

# f = data.iloc[0:2][['c1', 'c3']]    #选取区块数据，通常先通过iloc选取行，再选取列
# print(f)

"数据筛选"
# a = data[data['c1'] > 1]  #通过在中括号里设定筛选条件可以过滤行
# print(a)

# b = data[(data['c1'] > 1) & (data['c2'] < 8)]   #如果有多个筛选条件，可以通过'&'（表示"且"）或"｜"（表示"或 "）连接，"=="（表示俩者是否相等）
# print(b)


"数据整体情况查看"

#print(data.shape)   #通过dataframe的shape属性可以获取表格整体的行数和列数，从而快速了解表格数据量的大小

#print(data.describe())   #通过describe（）函数可以快速查看表格每一列的数据个数、平均值、标准差、最小值、25分位数、50分位数、75分位数、最大值等信息

#print(data['c1'].value_counts())    #通过value_counts（）函数可以快速查看某一列有几种数据，以及每一种数据出现的频次
"数据运算"
# data['c4'] = data['c3'] - data['c1']    #从已有的列中，通过数据运算创建新的一列
# print(data.head())

"数据排序"
# a = data.sort_values(by='c2', ascending=False)  #使用sort_values()函数可以对表格按列排序， 按列排序不写'by'效果一样，
# print(a)       #ascending参数默认为True，表示为升序，False为降序

# a = data.sort_values('c2', ascending=False)  #使用sort_index（）函数可以根据索引进行排序，如左侧按行索引进行升序排序
# a = a.sort_index()
# print(a)

"数据删除"
# a = data.drop(columns='c1')    #删除c1列的数据
# b = data.drop(columns=['c1', 'c3'])#删除多列的数据
# c = data.drop(index=['r1', 'r2'])   #删除多行数据   注意要输入行索引的名称而不是数字序号，除非行索引名称本来就是数字，才可以输入对应的数字
# data.drop(index=['r1', 'r2'], inplace=True)   #改变原表格data的结构，可以设置inplace参数为TRUE



