#1构造数据
import pandas as pd
data = pd.DataFrame([[22, 1], [25, 1], [20, 0], [35, 0], [32, 1], [38, 0], [50, 0], [46, 1]], columns=['年龄', '是否违约'])

#2数据分箱
data_cut = pd.cut(data['年龄'], 3)    #cut()函数进行等宽分箱，第一个参数待分箱等列，第二个参数是分箱个数

#3统计各个分箱的总样本数，坏样本数，和好样本数并汇总数据
#统计总客户数
cut_groupby_all = data['是否违约'].groupby(data_cut).count()
#统计违约客户数
cut_y = data['是否违约'].groupby(data_cut).sum()
#统计未违约客户数
cut_n = cut_groupby_all - cut_y
#汇总基础数据
df = pd.DataFrame()
df['总数'] = cut_groupby_all
df['坏样本'] = cut_y
df['好样本'] = cut_n
#4统计坏样本比率和好样本比率
df['坏样本%'] = df['坏样本'] / df['坏样本'].sum()
df['好样本%'] = df['好样本'] / df['好样本'].sum()
print(df)

#5计算WOE值
import numpy as np
df['WOE'] = np.log(df['坏样本%'] / df['好样本%'])
df = df.replace({'WOE': {np.inf: 0, -np.inf: 1}})   #替换可能存在的无穷大

#6计算各个分箱的IV值
df['IV'] = df['WOE'] * (df['坏样本%'] - df['好样本%'])

#7汇总各个分箱的IV值】
iv = df['IV'].sum()
print(iv)

