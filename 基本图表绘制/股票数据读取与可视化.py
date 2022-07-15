#第一步股票数据读取
import tushare as ts    #导入获取股价数据的库
import pandas as pd
import matplotlib.pyplot as plt
pro = ts.pro_api('bfecb1437a0b3b94122ed3b30e9c905e9c4802501a8890f3df2457ed')      #ts.pro_api(注册获得token填在这里)
#在Tushare pro 数据接口里，股票代码参数都叫ts_code，每种股票代码都有规范的后缀
#pro.daily（'股票代码'，start='起始日期'， end='结束日期'）函数获取上市公司万科十年的股票日线级别的数据
df = pro.daily(ts_code='000002.SZ', start_date='2009-01-01', end_date='2019-01-01')  #股票接口爬取
# df.head()  #head()函数是查看向量，矩阵或数据框等数据的部分信息,它默认输出数据框前6行数据,与其相对的是tail(),查看的是数据框最后的6行数据。
# df.to_excel('股价数据.xlsx', index=False)  #获取数据写入excel工作薄 （index = False：输出不显示 index (索引)值 True则反之）

#第二步绘制股价走势图
df.set_index('trade_date', inplace=True)  #set_index（日期的列名，显不显示索引）函数将日期设置为行索引
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS'] #用来正常显示中文
df['close'].plot(title='万科股价走势图')   #plot(title='万科股价走势图') 设置图标题
plt.show()

"方法二"

import tushare as ts
from datetime import datetime
#通过ts库获取股价数据
pro = ts.pro_api('bfecb1437a0b3b94122ed3b30e9c905e9c4802501a8890f3df2457ed')      #ts.pro_api(注册获得token填在这里)
df = pro.daily(ts_code='000002.SZ', start_date='2009-01-01', end_date='2019-01-01')  #股票接口爬取
#注意细节：调整日期格式，让横坐标的显示效果更清晰、美观
#df【'trade_date】是string字符串类型的，若直接用于画图，横坐标轴会显得很密集，影响美观
#所以这里通过datetime.strptime()函数将其转化为timestamp时间戳格式，这样Matplotlib库会自动间隔显示日期，其中apply（）函数与lambda函数
#联合使用是pandas库进行批处理的常见手段
df['trade_date'] = df['trade_date'].apply(lambda x:datetime.strptime(x, '%Y%m%d'))#%Y是选取年份
#绘制折线图plot
plt.plot(df['trade_date'], df['close'])
plt.show()


