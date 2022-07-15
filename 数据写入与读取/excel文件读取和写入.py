#excel文件写入#
#先创建一个Dataframe
import pandas as pd
data = pd.DataFrame([[1, 2], [3, 4], [5, 6]], columns=['A列', 'B列'])
#将Dataframe中的数据写入excel中
data.to_excel('data_new.xlsx')
print(data);

"excel文件读取方法一"

import pandas as pd
file = '/Users/macbookair/PycharmProjects/pythonProject1/venv/10月份-每月每日成交数据.xlsx'  #指定路径
f = open(file, 'rb')
df = pd.read_excel(f, sheet_name='每日有效投资人数')   #sheet_name：指定工作表名

f.close()  # 没有使用with的话，记得要手动释放。
# ------------- with模式 -------------------
with open(file, 'rb') as f:
    df = pd.read_excel(f, sheet_name='每日有效投资人数')


#df_dict = pd.read_excel(file, sheet_name='每日有效投资人数')  #sheet_name：指定解析名为"每日有效投资有效人数"的工作表。返回一个DataFrame类型的数据。
#df_dict = pd.read_excel(file, sheet_name='每日投资有效人数', header=None)   #如果不指定任何行作为列名，或数据源是无标题行的数据，可以显示的指定header=None来表明不使用列名。
#df_dict = pd.read_excel(file, sheet_name=[0,1,'每日投资有效人数'])#sheet_name=[0, 1, ‘Sheet1’], 对应的是解析文件的第1， 2张工作表和名为"Sheet1"的工作表。它返回的是一个有序字典。结构为{name：DataFrame}这种类型。
#df_dict = pd.read_excel(file, sheet_name=None)#sheet_name=None 会解析该文件中所有的工作表，返回一个同上的字典类型的数据。

df_dict = pd.read_excel(file, sheet_name='每日有效投资人数', header=0)    #header=0，用来指定数据的标题行，也就是数据的列名的
#df_dict = pd.read_excel(file, sheet_name='每日投资有效人数', header=None)     #如果不指定任何行作为列名，或数据源是无标题行的数据，可以显示的指定header=None来表明不使用列名。
#df = pd.read_excel(file, sheet_name='每日投资有效人数', names=list('123456789ABCDE'))   #names： 指定列名，指定数据的列名，如果数据已经有列名了，会替换掉原有的列名。
#df = pd.read_excel(file, sheet_name='每日有效投资人数', names=list('123456789ABCDE'), header=None)  #header=0默认第一行中文名是标题行，最后被names给替换了列名，如果只想使用names，而又对源数据不做任何修改，我们可以指定header=None

df = pd.read_excel(file, sheet_name='每日有效投资人数', header=0, index_col=0)   #index_col: 指定列索引
df = pd.read_excel(file, sheet_name='每日有效投资人数', skiprows=0)   #skiprows：跳过指定行数的数据
#df = pd.read_excel(file, sheet_name='每日有效投资人数', skiprows=[1,3,5,7,9,])  ##skiprows：跳过指定行数[1,3,5,7,9,]的数据
#df = pd.read_excel(file, sheet_name='每日有效投资人数', skipfooter=5)    #skipfooter：省略从尾部的行数据，从尾部跳过5行

#df = pd.read_excel(file, sheet_name='每日有效投资人数', header=0, dtype={'当日有效投资人数': str})  #dtype 指定某些列的数据类型，指定当日有效投资人数列的数据类型：
def date(para):    #转换日期的格式
    if type(para) == int:
        delta = pd.Timedelta(str(int(para))+'days')
        time = pd.to_datetime('1899-12-30') + delta
        return time
    else:
        return para
df['日期'] = df['日期'].apply(date)
df
print(df)




