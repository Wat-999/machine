# # 8.3 量化金融 - 股票涨跌预测模型搭建
# # 8.3.1 多因子模型搭建
# **1.引入之后需要用到的库**
import tushare as ts
import numpy as np
import pandas as pd
import talib
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings("ignore")

# **2.股票数据处理与衍生变量生成**
# 1.股票基本数据获取
pro = ts.pro_api('bfecb1437a0b3b94122ed3b30e9c905e9c4802501a8890f3df2457ed')      #ts.pro_api(注册获得token填在这里)
df = pro.daily(ts_code='000002.SZ', start_date='20150101', end_date='20191231')#pro.daily（'股票代码'，start='起始日期'， end='结束日期'）函数获取上市公司万科十年的股票日线级别的数据
#https://tushare.pro/user/token

df = df.sort_index()   #sort_index() 将数据修改按日期远到近排序
df = df.set_index('trade_date')



# 2.简单衍生变量构造
df['close-open'] = (df['close'] - df['open'])/df['open']
df['high-low'] = (df['high'] - df['low'])/df['low']

df['pre_close'] = df['close'].shift(1)
df['price_change'] = df['close']-df['pre_close']
df['p_change'] = (df['close']-df['pre_close'])/df['pre_close']*100

# 3.移动平均线相关数据构造
df['MA5'] = df['close'].rolling(5).mean()
df['MA10'] = df['close'].rolling(10).mean()
df.dropna(inplace=True)  # 删除空值

# 4.通过Ta_lib库构造衍生变量
df['RSI'] = talib.RSI(df['close'], timeperiod=12)
df['MOM'] = talib.MOM(df['close'], timeperiod=5)
df['EMA12'] = talib.EMA(df['close'], timeperiod=12)
df['EMA26'] = talib.EMA(df['close'], timeperiod=26)
df['MACD'], df['MACDsignal'], df['MACDhist'] = talib.MACD(df['close'], fastperiod=12, slowperiod=26, signalperiod=9)
df.dropna(inplace=True)

# **3.特征变量和目标变量提取**
X = df[['close', 'vol', 'close-open', 'MA5', 'MA10', 'high-low', 'RSI', 'MOM', 'EMA12', 'MACD', 'MACDsignal', 'MACDhist']]
y = np.where(df['price_change'].shift(-1)> 0, 1, -1)

# **3.训练集和测试集数据划分**
X_length = X.shape[0]  # shape属性获取X的行数和列数，shape[0]即表示行数
split = int(X_length * 0.9)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# **4.模型搭建**
model = RandomForestClassifier(max_depth=3, n_estimators=10, min_samples_leaf=10, random_state=1)
model.fit(X_train, y_train)
# # 8.3.2 模型使用与评估
# **1.预测下一天的涨跌情况**
y_pred = model.predict(X_test)
print(y_pred)

a = pd.DataFrame()  # 创建一个空DataFrame
a['预测值'] = list(y_pred)
a['实际值'] = list(y_test)
a.head()

y_pred_proba = model.predict_proba(X_test)
#print(y_pred_proba[0:5])

# **2.模型准确度评估**
from sklearn.metrics import accuracy_score
score = accuracy_score(y_pred, y_test)
#print(score)
#print(model.score(X_test, y_test))

# **3.分析数据特征的重要性**
#print(model.feature_importances_)

features = X.columns
importances = model.feature_importances_
a = pd.DataFrame()
a['特征'] = features
a['特征重要性'] = importances
a = a.sort_values('特征重要性', ascending=False)
#print(a)

# # 8.3.4 收益回测曲线绘制
X_test['prediction'] = model.predict(X_test)
X_test['p_change'] = (X_test['close'] - X_test['close'].shift(1)) / X_test['close'].shift(1)
X_test['origin'] = (X_test['p_change'] + 1).cumprod()
X_test['strategy'] = (X_test['prediction'].shift(1) * X_test['p_change'] + 1).cumprod()
print(X_test[['strategy', 'origin']].tail)

X_test[['strategy', 'origin']].dropna().plot()
plt.gcf().autofmt_xdate()
plt.show()
