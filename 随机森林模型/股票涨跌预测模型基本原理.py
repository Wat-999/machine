#1获取股票基本数据
import tushare as ts
pro = ts.pro_api('bfecb1437a0b3b94122ed3b30e9c905e9c4802501a8890f3df2457ed')      #ts.pro_api(注册获得token填在这里)
df = pro.daily(ts_code='000002.SZ', start_date='20150101', end_date='20191231')#pro.daily（'股票代码'，start='起始日期'， end='结束日期'）函数获取上市公司万科十年的股票日线级别的数据
#https://tushare.pro/user/token
df.head()

df = df.set_index('trade_date')
df = df.sort_index()   #sort_index() 将数据修改按日期远到近排序
df.head()
#print(df.head())

#2生成简单衍生变量
df['close-open'] = (df['close'] - df['open'])/df['open']  #表示（收盘价-开盘价）/开盘价
df['high-low'] = (df['high'] - df['low'])/df['low']   #表示（最高价-最低价）/最低价
df['pre_close'] = df['close'].shift(1)     #pre_close表示昨日收盘价格，用shift（1）将close列所有数据向下移动一行并形成新的一列，如果是shift（-1）则表示向上移动1行
df['price_change'] = df['close'] - df['pre_close']    #表示今日收盘价-昨日收盘价
df['p_change'] = (df['close'] - df['pre_close'])/df['pre_close']*100   #表示当天股价变化的百分比，也称当天股价的涨幅

#print(df.head())
#3生成移动平均线指标MA值
df['MA5'] = df['close'].rolling(5).mean()
df['MA10'] = df['close'].rolling(10).mean()
df = df.dropna()   #删除空值行，也可以写成df.dropna(inplace=True)
df.head()

#4用TA-Lib库生成相对强弱指标RSI值
import talib
import pandas as pd
df['RSI'] = talib.RSI(df['close'], timeperiod=12)
data = pd.DataFrame()
data['close'] = [10, 12, 11, 13, 12, 14, 13]
data['RSI'] = talib.RSI(data['close'], timeperiod=6)
#print(data.head(7))
#RSI值解说
#RSI值能反映短期股价涨势相对跌势但强弱，帮助我们更好地判断股价但涨跌趋势，RSI值越大，涨势相对于跌势越强，反之涨势相对于跌势越弱
#RSI=（N日平均上涨价格）/（N日平均上涨价格+N日平均下跌价格） *100%  前面代码设置timeperiod参数为12，即N取12
#通常情况下，RSI值位于20～80之间，超过80则为超买状态，低于20则为超卖状态，等于50则认为买卖双方力量均等
#例如，如果连续6天股价都是上涨，则6日平均下跌价格为0，6日RSI值为100，表明此时股票买方处于非常强势的地位，但也提醒投资者要警惕此时可能也是超买状态，需要预防股价下跌但风险

#5用talib库生成动量指标MOM值
df['MOM'] = talib.MOM(df['close'], timeperiod=5)  #MOM是动量的缩写：C-Cn  其中c表示当前的收盘价，Cn表示n天前的收盘价
#假设要计算6号的MOM值，而设置参数timeperiod=5，那么就需要用6号的收盘价减去1号的收盘价，后面的同理，再将连续几天的MOM值连起来就构成一条反映股价涨跌变动的曲线

#6用talib库生成指数移动平均值EMA
df['EMA12'] = talib.EMA(df['close'], timeperiod=12)   #12日指数移动平均值（快）
df['EMA26'] = talib.EMA(df['close'], timeperiod=26)   #26日指数移动平均值（慢）
#EMA是以指数式递减加权的移动平均，并根据计算结果进行分析，用于判断股价未来走势的变动趋势。EMA和移动平均线指标MA值有点类似
#详细计算方法参考书178页
#通过talib库可以验证上面的计算结果
data = pd.DataFrame()
data['close'] = [1, 1, 1, 1, 1, 7, 2]
data['EMA6'] = talib.EMA(data['close'], timeperiod=6)
#print(data)
#计算公式：EMA（today）= αPrice（taoday）+（1-α）EMA(yesterday）
#EMA（today）为当天的EMA值，Price（taoday）为当天的收盘价，EMA(yesterday）为昨天的EMA值，α为平滑指数，一般取值为2/（N+1），N表示为天数
#当N为6时，α为2/7，对应的EMA称为EMA6，即6日指数移动平均值。公式不断递归，直至第一个EMA值出现（第一个EMA值通常为开头6个数的均值）
#数据为（#通过talib库可以验证上面的计算结果）演示EMA6计算：取第一个EMA值为开头6个数的均值，故前5天都没有EMA值，6号的EMA值就是第一个EMA值，为前6天的均值，即为2
#7号的EMA值为第二个EMA值，计算为：EMA（today）=2/7*2+（1-2/7）*EMA(yesterday）=2/7*2+5/7*2=2
#对于EMA值而言，近期的股价比之前更久远的股价更重要（在其计算公式中，近期股权的权重更大），不想MA值那样一视同仁


#7用talib库生成异同移动平均线MACD值
df['MACD'], df['MACDsignal'], df['MACDhist'] = talib.MACD(df['close'], fastperiod=6, slowperiod=12, signalperiod=9)
df.dropna(inplace=True)
#MACD是股票市场上的常用指标，它是基于EMA值的衍生变量简单来说MACD是从双指数移动平均值发展而来的，由快的指数移动平均线EMA12减去慢的指数移动平均线EMA26得到的DIF值
#再用2*（DIF值-DIF值的9日加权移动均线DEA值）得到MACD值
#1计算EAM12和EMA26他们的计算步骤一致，不同的是计算公式中的α分别为2/13和2/27
#2计算DIF值和其9日加权移动均线DEA值
#DIF值就是EMA12-EMA26,反映的是俩条移动平均线的差值。DIF值对应的连线也称为快速线
#DEA值是DIF值的9日加权移动均线值，计算方法类似EMA9，区别就是计算EMA9用的是收盘价，而计算DEA值用的是DIF值。DEA值对应的连线也称为慢速线
#3计算MACD值MACD值就是2*（DIF-DEA)，也称为MACD柱。因此MACD技术指标图是由两线一柱组合而成的，快速线为DIF值，慢速线为DEA值，柱形图为MACD值
#当MACD值大于0时（在图形中反映为0轴上方的红柱），DIF值大于DEA值，即快速线在慢速线之上表示行情正处于上涨的走势。
#其实这个分析思路和移动平均线MA值的分析思路有点类似：当MA5>MA10时，也表示行情正处于上涨的走势，不过MACD值经过一系列加权，会更严谨。
#反之，当MACD值小于0时（在图形中反映为0轴下方的绿柱），则表示行情正处于下跌的走势
#当DIF值和DEA值均大于0时并向上移动时，一般表示行情正处于多头的行情中可以买入开仓或多头持仓；当DIF值和DEA值均小于0并向下移动时，
#一般表示行情正处于空头行情中，可以卖出开仓或观望；当DIF值和DEA值均大于0但都向下移动时，一般表示行情正处于下跌阶段，可以卖出开仓或观望；
#当DIF值和DEA值均小于0时但都向上移动时，一般表示行情即将上涨，可以买入开仓或多头持仓。

#了解MACD的基础知识后，下面补充说明TA-Lib库中MACD、MACDsignal、MACDhist的定义。因为TA-Lib库是由国外的程序员开发的，所以其中一些值的定义和
#国内的定义略有不同，不过基本思路是一致的：MACD对应国内的DIF值，MACDsignal对应国内的DEA值，MACDhist对应国内MACD值的一半，即DIF值-DEA值