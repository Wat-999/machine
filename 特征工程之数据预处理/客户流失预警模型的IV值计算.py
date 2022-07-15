#为了提高代码通用性，写成自定义函数形式
#该函数共有4个参数data(原始数据集）、cut_num(数据分箱步骤中分箱的个数）、feature(需要计算IV值的特征变量名称）、target(目标变量名称）
#有了这个函数，就能方便地对任意一个数据集计算各个特征变量的IV值
import pandas as pd
import numpy as np

def cal_iv(data, cut_num, feature, target):
    #1数据分箱
    data_cut = pd.cut(data[feature], cut_num)

    #2统计各个分箱的总样本数、坏样本数和好样本数
    cut_group_all = data[target].groupby(data_cut).count()  #统计总样本数
    cut_y = data[target].groupby(data_cut).sum()   #统计坏样本数
    cut_n = cut_group_all - cut_y     #统计好样本数

    #汇总基础数据
    df = pd.DataFrame()
    df['总数'] = cut_group_all
    df['坏样本'] = cut_y
    df['好样本'] = cut_n

    #3统计坏样本比率和好样本比率
    df['坏样本%'] = df['坏样本'] / df['坏样本'].sum()
    df['好样本%'] = df['好样本'] / df['好样本'].sum()

    #4计算WOE值
    df['WOE'] = np.log(df['坏样本%'] / df['好样本%'])
    df = df.replace({'WOE': {np.inf: 0, -np.inf: 0}})

    #5计算各个分箱的IV值
    df['IV'] = df['WOE'] * (df['坏样本%'] - df['好样本%'])

    #6汇总各个分箱的IV值，获得特征变量的IV值
    iv = df['IV'].sum()

    print(iv)

#读取数据
data = pd.read_excel('/Users/macbookair/Desktop/数据分析/书本配套资料及电子书/python/Python大数据分析与机器学习/源代码汇总-2020-12-16/第11章 特征工程之数据预处理/源代码汇总_PyCharm格式/股票客户流失.xlsx')

#利用创建好的自定义函数计算第一个特征变量"账户资金（元）"的IV值
cal_iv(data, 4, '账户资金（元）', '是否流失')

#通过for循环可以快速计算出所有特征变量的IV值
for i in data.columns[:-1]:   #data.columns用于获取所有的列名，因为考量的是特征变量，不需要最后一列目标变量"是否流失"（切片左开右闭）
    print(i + '的IV值为：')
    cal_iv(data, 4, i, '是否流失')       #调用函数

#从打印结果可得出结论："本券商使用时长（年）"的信息量最大，而"账户资金（元）"的信息量最小，预测能力最低。这其实也是搭建逻辑回归模型时判断特征重要性的一个方式