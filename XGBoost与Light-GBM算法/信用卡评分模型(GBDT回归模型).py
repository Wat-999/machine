#1读取数据
import numpy as np
import pandas as pd
df = pd.read_excel('/Users/macbookair/Desktop/数据分析/书本配套资料及电子书/python/Python大数据分析与机器学习/源代码汇总-2020-12-16/第10章 机器学习神器：XGBoost&LightGBM模型/源代码汇总_Pycharm/信用评分卡模型.xlsx')

#2提取特征变量和目标变量
X = df.drop(columns='信用评分')
y = df['信用评分']

#3划分训练集和测试集
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

#4模型训练和搭建
from xgboost import XGBRegressor
model = XGBRegressor()   #构建GBDT回归模型
model.fit(X_train, y_train)

#5模型预测及评估
#对测试集数据进行预测
y_pred = model.predict(X_test)
print(y_pred)

#汇总预测值和实际值
a = pd.DataFrame()
a['预测值'] = list(y_pred)
a['实际值'] = list(y_test)
print(a)

#查看R-squared值来评价模型的拟合效果
from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_pred)  #将测试集的实际值和模型的预测值传入
#model.score(X_test, y_test)   #等同r2
print('R-squared值:' + str(r2))  #R-squared值:0.676
#这个结果比多元线性回归模型获得的0.629相比是有所改善的


#查看特征重要性
features = X.columns   #获取特征名称
importances = model.feature_importances_     #获取特征重要性
#整理成二维表格，并按特征重要性降序排列
importances_df = pd.DataFrame()
importances_df['特征名称'] = features
importances_df['特征重要性'] = importances
print(importances_df.sort_values('特征重要性', ascending=False))
#结果解读，可以看到'月收入'的特征重要性最高，'性别'的特征重要性最低。'历史违约次数'的特征重要性不是很高，这与经验认知不符合，
#可能的原因是数据较少，导致各个样本的'历史违约次数'相差不大，而实际应用中当数据量较大时，该特征还是有较高的特征重要性。

#模型参数调优
from sklearn.model_selection import GridSearchCV
import numpy as np
parameters = {'max_depth': np.arange(1, 6, 2), 'n_estimators': np.arange(50, 200, 50), 'learning_rate': [0.01, 0.05, 0.1, 0.2]}
#参数：learning_rate     含义：弱学习器的权重缩减系数   取值：取之范围为(0,1],取值较小意味着达到一定的误分类数或学习效果需要更多迭代次数和更多弱学习器，默认取值0.1,即不缩减
clf = XGBRegressor()  # 构建回归模型
grid_search = GridSearchCV(model, parameters, scoring='r2', cv=5)
#scoring为设置模型评估标准,注意：因为XGBRegressor是回归模型，所以参数调优时应该选择R-squared值作为评估标准而不是分类模型中常用的准确度(accuracy)和ROC中的AUC值

grid_search.fit(X_train, y_train)   #传入训练集数据
print(grid_search.best_params_)     #输出参数的最优值  'learning_rate': 0.1, 'max_depth': 3, 'n_estimators': 50}

#用参数的最优值搭建新模型
model = XGBRegressor(max_depth=3, n_estimators=50, learning_rate=0.1)
model.fit(X_train, y_train)
#进行调优后模型评估
from sklearn.metrics import r2_score
r2 = r2_score(y_test, model.predict(X_test))
print('调优后R-squared：' + str(r2))


# **补充知识点2：对于XGBoost模型，有必要做很多数据预处理吗？**
#对数据进行Z—score标准化
from sklearn.preprocessing import StandardScaler
X_new = StandardScaler().fit_transform(X)
print(X_new)   #此时获得的X_new是一个二维数组，可以看到这些特征都变成以0为均值、1为标准差的数据分布来

# 3.划分测试集和训练集
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_new, y, test_size=0.2, random_state=123)

# 4.建模
from xgboost import XGBRegressor
model = XGBRegressor()
model.fit(X_train, y_train)

from sklearn.metrics import r2_score
r2 = r2_score(y_test, model.predict(X_test))
print(r2)  #与没有进行特征变换，数据标准化值一样

#在传统的机器学习模型中，往往需要做很多数据预处理工作，如数据的标准化、缺失值和异常常值的处理。但对于XGBoost模型而言
#很多但预处理都不是必要的，例如缺失值，XGBoost模型会自动处理，它会通过列举所有缺失值在当前节点是进入左子树还是右子树来决定缺失值的处理方式
#此外，因为XGBoost模型是基于决策树模型的，所以像线性回归等模型需要的特征变换(如离散化、标准化、取log、共线问题处理）等预处理工作，
#XGBoost模型都不太需要，这也是树模型优点之一。如果还不太放心，可以尝试进行特征变化，如数据标准化，会最终发现的结果都是一样的。
#这里通过对X_new运用train_test_split()函数划分训练集和测试集合，并进行模型训练，最后用r2_score()函数计算模型评分，会发现结果和数据没有标准化
#结果一样，都为0.572,这也验证了树模型不需要进行特征的标准化。此外，树模型对于共线性也不敏感
#绝大部分模型不能自动完成的一步就是特征提取。很多自然语言处理或图像处理的问题，没有现成的特征，需要人工去提取特征。
#综上所述，XGBoost模型的确比线性模型要节省很多特征工程的步骤，但特征工程依然是非常必要的，这一结论同样适用LightGBM模型