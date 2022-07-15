#1引入需要的库
import tushare as ts  #引入股票基本数据相关库
import numpy as np    #引入科学计算相关库
import pandas as pd   #引入科学计算相关库
import talib          #引入股票衍生变量数据相关库
import matplotlib.pylab as plt   #引入绘图相关库
from sklearn.ensemble import RandomForestClassifier     #引入分类决策树模型
from sklearn.metrics import accuracy_score              #引入预测准确度评分函数

#2获取数据
#1获取股票基本数据
pro = ts.pro_api('bfecb1437a0b3b94122ed3b30e9c905e9c4802501a8890f3df2457ed')      #ts.pro_api(注册获得token填在这里)
df = pro.daily(ts_code='000002.SZ', start_date='20150101', end_date='20191231')#pro.daily（'股票代码'，start='起始日期'， end='结束日期'）函数获取上市公司万科十年的股票日线级别的数据
#https://tushare.pro/user/token

df = df.sort_index()   #sort_index() 将数据修改按日期远到近排序
df = df.set_index('trade_date')
#print(df.head())

#2生成简单衍生变量
df['close-open'] = (df['close'] - df['open'])/df['open']  #表示（收盘价-开盘价）/开盘价
df['high-low'] = (df['high'] - df['low'])/df['low']   #表示（最高价-最低价）/最低价
df['pre_close'] = df['close'].shift(1)     #pre_close表示昨日收盘价格，用shift（1）将close列所有数据向下移动一行并形成新的一列，如果是shift（-1）则表示向上移动1行
df['price_change'] = df['close'] - df['pre_close']    #表示今日收盘价-昨日收盘价
df['p_change'] = (df['close'] - df['pre_close'])/df['pre_close']*100   #表示当天股价变化的百分比，也称当天股价的涨幅
#print(df.head())

#3生成移动平均线指标MA值
df['MA5'] = df['close'].rolling(5).mean()    #5日移动平均线
df['MA10'] = df['close'].rolling(10).mean()  #10日移动平均线
df.dropna(inplace=True)  #删除空值行，也可以写成df.dropna()
df.head()

#4通过TA-Lib库构造衍生变量数据
df['RSI'] = talib.RSI(df['close'], timeperiod=12)
df['MOM'] = talib.MOM(df['close'], timeperiod=5)
df['EMA12'] = talib.EMA(df['close'], timeperiod=12)
df['EMA26'] = talib.EMA(df['close'], timeperiod=26)
df['MACD'], df['MACDsignal'], df['MACDhist'] = talib.MACD(df['close'], fastperiod=12, slowperiod=26, signalperiod=9)
df.dropna(inplace=True)
#print(df.tail())

#3提取特征变量和目标变量
#选择特征变量（也称因子）赋给变量X
X = df[['close', 'vol', 'close-open', 'MA5', 'MA10', 'high-low', 'RSI', 'MOM', 'EMA12', 'MACD', 'MACDsignal', 'MACDhist']]
#构造目标变量y，这里使用了numpy库中where()函数，传入三个参数的含义分别为判断条件、满足条件的赋值、不满足条件的赋值
#shift(-1)是将'price_change'股价变化这一列所有数据向上移动一行，获得每一行对应的下一天股价变化
#因此这里的判断条件就是下一天的股价变化是否大于0，如果大于0，说明下一天股价涨了，则y赋值为1，如果不大于0，说明下一天不变或跌了，则y赋值为-1。预测结果就只有1和-1两种分类
y = np.where(df['price_change'].shift(-1)>0, 1, -1)

#4划分训练集和测试集
#注意：划分要按照时间序列进行，而不能用train_test_split()函数进行随机划分。因为股价的变化趋势具有时间特征，而随机划分会破坏这种特征
#所以需要根据当天股价数据预测下一天的股价涨跌情况
X_length = X.shape[0]        #用shape属性获取X行数和列数，shape[0]即为行数
split = int(X_length * 0.9)                #将前90%的数据划分为训练集，后10%的数据作为测试集
X_train, X_test = X[:split], X[split:]
y_train , y_test = y[:split], y[split:]


#5模型搭建
model = RandomForestClassifier(max_depth=3, n_estimators=10, min_samples_leaf=10, random_state=1)
#max_depth设置为3，即决策树的最大深度为3，即每个决策树最多只有3层
#弱学习器，即决策树模型的个数n_estimators设置为10，即该随机森林中共有10个决策树
#叶子节点的最小样本数min_samples_leaf设置为10，即如果叶子节点的样本数小于10则停止分裂，
#随机状态参数random_state的作用是使每一次运行结果保持一致，这里设置的数字1没有特殊含义，可以换成其他数字
model.fit(X_train, y_train)     #传入训练集参数

#6模型使用与评估
#1预测下一天的股价涨跌情况
y_pred = model.predict(X_test)    #传入测试集参数
print(y_pred)
#汇总预测值y_pred和测试集的实际值y_test
a = pd.DataFrame()
a['预测值'] = list(y_pred)
a['实际值'] = list(y_test)
#print(a.head())

#2用predict_proba()函数预测属于各个分类的概率
y_pred_proba = model.predict_proba(X_test)   #注意：y_pred_proba是一个二维数组
b = pd.DataFrame(y_pred_proba, columns=['分类为-1的概率', '分类为1的概率'])
#print(b.head())

#3模型准确度评估
#查看整体的预测准确度
score =accuracy_score(y_pred, y_test)   #传入预测值与实际值
#core = model.score(X_test, y_test)     #传入测试集参数 与上面结果一致
#print("整体预测准确度：" + str(score))
#说明模型对整个测试集合中约90%的数据预测准确，整体预测准确度算挺不错的，但在商业实战中，评估模型不会只看其预测准测度，更着重收益回测曲线绘制情况

#4分析特征变量的特征重要性（model.feature_importances_）
features = X.columns
importtances = model.feature_importances_
a = pd.DataFrame()
a['特征'] = features
a['特征重要性'] = importtances
a = a.sort_values('特征重要性', ascending=False)
#print(a)

#7参数调优
from sklearn.model_selection import GridSearchCV
parameters = {'n_estimators':np.arange(5, 25, 5), 'max_depth':np.arange(2, 6, 1), 'min_samples_leaf':np.arange(5, 35, 5)}
#'n_estimators'为弱学习器(决策树的个数）默认为10个(int型）
#'max_depth'为弱学习器(决策树的最大深度)取值为int型数据或None，如果为None，则会扩展所有节点，直到所有叶子节点是纯净的
#'min_samples_leaf'为叶子节点的最小样本数
new_model = RandomForestClassifier(random_state=1)   #构建新模型
grid_search = GridSearchCV(new_model, parameters, cv=6, scoring='accuracy')
#cv参数设置为6表示交叉验证6次，设置模型评估标准scoring参数为'accuracy'，即以准确度作为评估标准，也可以设置成'roc_auc'则表示以ROC曲线的AUC值作为评估标准
grid_search.fit(X_train, y_train)
c = grid_search.best_params_
#print(c)   #参数调优结果：{'max_depth': 2, 'min_samples_leaf': 25, 'n_estimators': 15}  再重新更换模型参数，结果更优

#8收益回测曲线绘制
#前面已经评估了模型的预测准确度，不过在商业实战中，更关心它的收益回测曲线（又称为净值曲线），也就是看根据搭建的模型获得的结果是否比不利用模型获得的结果更好
X_test['prediction'] = model.predict(X_test)   #获取预测值（根据当天的股价数据预测下一天的股价数据预测下一天的股价涨跌情况）
#计算每天的股价变化率即（当天的收盘价-前一天的收盘价）/前一天的收盘价
X_test['p_change'] = (X_test['close'] - X_test['close'].shift(1)) / X_test['close'].shift(1)
#计算原始数据的收益率，这里主要用到了累乘函数cumprod（）。初始股价是1，2天内的价格变化率为10%，那么用cumprod（）函数可以求得2天后的股价为1*1（1+10%）*（1+10%）=1.21,此结果也表明2天收益2天的收益率21%
X_test['origin'] = (X_test['p_change'] + 1).cumprod()

#计算利用模型预测后的收益率
X_test['strategy'] = (X_test['prediction'].shift(1) * X_test['p_change'] + 1).cumprod()
print(X_test[['strategy', 'origin']].tail)

#可视化
X_test[['strategy', 'origin']].dropna().plot()
plt.gcf().autofmt_xdate()
plt.show()