#1读取数据
import pandas as pd
df = pd.read_excel('/Users/macbookair/Desktop/数据分析/书本配套资料及电子书/python/Python大数据分析与机器学习/源代码汇总-2020-12-16/第10章 机器学习神器：XGBoost&LightGBM模型/源代码汇总_Pycharm/信用卡交易数据.xlsx')

#2提取特征变量和目标变量
X = df.drop(columns='欺诈标签')
y = df['欺诈标签']

#3划分训练集和测试集
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

#4模型训练及搭建
from xgboost import XGBClassifier    #引入xgboost分类模型XGBClassifier
clf = XGBClassifier(n_estimators=100, learning_rate=0.05)   #第一个参数表示弱学习器个数为100，第二个参数表示弱学习器的权重缩减系数为0.05
clf.fit(X_train, y_train)

#5模型预测及评估
#对测试集数据进行预测
y_pred = clf.predict(X_test)
print(y_pred)

#汇总预测值和实际值
a = pd.DataFrame()
a['预测值'] = list(y_pred)
a['实际值'] = list(y_test)
print(a.head())

#测试集数据的预测准确度
from sklearn.metrics import accuracy_score
score = accuracy_score(y_pred, y_test)
print('预测准确度：' + str(score))

#预测属于各个分类的概率
y_pred_proba = clf.predict_proba(X_test)
print(y_pred_proba[:, 1])

#绘制roc曲线来评估模型的预测效果
from sklearn.metrics import roc_curve
fpr, tpr, thers = roc_curve(y_test, y_pred_proba[:, 1])
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS'] #用来正常显示中文
plt.plot(fpr, tpr)
plt.title('ROC')
plt.xlabel('假警报率')
plt.ylabel('命中率')
plt.show()

#计算模型的ACU值
from sklearn.metrics import roc_auc_score
score = roc_auc_score(y_test, y_pred_proba[:, 1])
print('AUC:' + str(score))

#分析数据特征的重要性，并对特征名称和特征重要性进行汇总
features = X.columns  # 获取特征名称
importances = clf.feature_importances_  # 获取特征重要性

importances_df = pd.DataFrame()
importances_df['特征名称'] = features
importances_df['特征重要性'] = importances
importances_df.sort_values('特征重要性', ascending=False)
print(importances_df.sort_values('特征重要性', ascending=False))

#6模型参数调优
from sklearn.model_selection import GridSearchCV
#设定模型的超参数
parameters = {'max_depth': [1, 3, 5], 'n_estimators': [50, 100, 150], 'learning_rate': [0.01, 0.05, 0.1, 0.2]}  # 指定模型中参数的范围
#参数：learning_rate     含义：弱学习器的权重缩减系数   取值：取之范围为(0,1],取值较小意味着达到一定的误分类数或学习效果需要更多迭代次数和更多弱学习器，默认取值0.1,即不缩减
clf = XGBClassifier()  # 构建模型
grid_search = GridSearchCV(clf, parameters, scoring='roc_auc', cv=5) #以roc曲线的ACU值作为模型的评估标准

grid_search.fit(X_train, y_train)  # 传入数据
print(grid_search.best_params_)  # 输出参数的最优值

#使用上面获得的参数最优值重新搭建模型
clf = XGBClassifier(max_depth=1, n_estimators=100, learning_rate=0.05)  #传入模型的超参数
clf.fit(X_train, y_train)

y_pred_proba = clf.predict_proba(X_test)
from sklearn.metrics import roc_auc_score
score = roc_auc_score(y_test, y_pred_proba[:, 1])
print('调优后ACU的值：' + str(score))

#结果解读，获得的AUC值为0.856,略低于调优前0.868。为什么调优后的结果还不如调优前的结果？通常来说，出现这种情况的概率较小，
#可能是因为数据两较小(如本案例只有1000条数据),也可能是因为本案例中使用了交叉验证。简单回顾一下K折交叉验证的思路：
#将原来的样本数据分为k份(这里设置cv为5，即分为5份），然后在k份数据中选取k-1份作为训练集数据，剩下的一份作为测试数据，训练k次
#获得k个roc曲线的AUC值，然后对k个AUC值求均值，再取AUC值的均值最大时的参数为模型的最优参数。
#需要注意的是，本案例中AUC值的获取是基于训练集数据，只不过是将训练集数据中的1/k作为测试集数据，该测试集数据并不是真正的测试集数据y_test
#这也是参数调优后的结果不如调优前的结果的原因。

