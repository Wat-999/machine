import tushare as ts
import mplfinance as mpf
from pylab import mpl
import pandas as pd
pro = ts.pro_api('9d674d000f7c730dd3108701a1a1c534bf51bfb03a0ff169a9d11848')      #ts.pro_api(注册获得token填在这里)  #有2100积分
df = pro.daily(ts_code='000001.SZ', start_date='20200101', end_date='20201103')#pro.daily（'股票代码'，start='起始日期'， end='结束日期'）函数获取上市公司万科十年的股票日线级别的数据
#https://tushare.pro/user/token

#df.sort_values(by='trade_date',ascending=False)
#取所有行数据，后面取date列，open列等数据   loc(取行（：表示所有行），取列)
data = df.loc[:, ['trade_date', 'open', 'close', 'high', 'low', 'vol']]
#rename函数：df.rename(columns = {'a':'aa'},inplace = True)加上inplace = True 才可以在原df上进行修改，可修改索引，也可修改列名
data = data.rename(columns={'trade_date': 'Date', 'open': 'Open', 'close': 'Close', 'high': 'High', 'low': 'Low', 'vol': 'Volume'})  #更换列名，为后面函数变量做准备

#设置date列为索引，覆盖原来索引,这个时候索引还是 object 类型，就是字符串类型。
data.set_index('Date', inplace=True)
#将object类型转化成 DateIndex 类型，pd.DatetimeIndex 是把某一列进行转换，同时把该列的数据设置为索引 index。
data.index = pd.DatetimeIndex(data.index)

#将时间顺序升序，符合时间序列
data = data.sort_index(ascending=True)

# pd.set_option()就是pycharm输出控制显示的设置，下面这几行代码其实没用上，暂时也留在这儿吧
pd.set_option('expand_frame_repr', False)#True就是可以换行显示。设置成False的时候不允许换行
pd.set_option('display.max_columns', None)# 显示所有列
#pd.set_option('display.max_rows', None)# 显示所有行
pd.set_option('colheader_justify', 'centre')# 显示居中


#mpl.rcParams['font.sans-serif'] = ['Arial Unicode MS'] #用来正常显示中文
mpl.rcParams["figure.figsize"] = [6.4, 4.8]   #设置图表大小
#mpl.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题
#s = mpf.make_mpf_style(rc={'font.family':'Arial Unicode MS'})   #改用样式来指定

#第二步，我们先使用一下mpf.make_marketcolors()函数，来设定一下K线的颜色方面的信息。
# 一会儿要把这个设定的结果作为实参传给mpf.make_mpf_style()来设定自定义的风格样式。
mc = mpf.make_marketcolors(
    up="red",  # 上涨K线的颜色
    down="green",  # 下跌K线的颜色
    edge="black",  # 蜡烛图箱体的颜色
    volume="blue",  # 成交量柱子的颜色
    wick="black"  # 蜡烛图影线的颜色
)

# 还有一个叫alpha的参数，设置的是candlestick face，取值在0.1-1之间。这个设置的是K线蜡烛颜色的深浅，比如把当alpha设置为0.6的时候红色蜡烛就变成了接近橘黄色。绿色就变成了翠绿色。这个根据自己的感官来尝试选择就好啦。

#mc设置好后，接下来我们要将其传给mpf.make_mpf_style()的marketcolors参数，来设定自定义的风格样式了
#第三步，我们开始设定自定义的风格样式了。
#使用mpf.make_mpf_style函数，其参数有：
s = mpf.make_mpf_style(
    gridaxis='both',   #设置网格线方向,both双向 'horizontal’水平, 'vertical’垂直
    gridstyle='-.',    #设置网格线线型
    y_on_right=True,   #设置y轴位置是否在右
    marketcolors=mc,   #定义的那个K线的属性，把它传入就OK了
    edgecolor='b',     #设置框线样式
    figcolor='r',      #设置图像外周边填充色
    facecolor='y',     #设置前景色（坐标系颜色）
    gridcolor='c',   #设置网格线颜色
    rc={'font.family':'Arial Unicode MS',   #用来正常显示中文
        'axes.unicode_minus': 'False'})     ## 解决保存图像是负号'-'显示为方块的问题

#第四步，开始使用mpf.plot()绘图了，传入上边设定好的风格s
mpf.plot(data,
         type='candle',
         title='万科k线图',
         ylabel="价格",
         ylabel_lower="成交量",
         mav=(5, 10, 20),
         volume=True,
         show_nontrading=False,
         style=s)  #应用样式style=s


