#1读取数据
import pandas as pd
df = pd.read_excel('/Users/macbookair/Desktop/数据分析/书本配套资料及电子书/python/Python大数据分析与机器学习/源代码汇总-2020-12-16/第9章 AdaBoost与GBDT模型/源代码汇总_Pycharm/信用卡精准营销模型.xlsx')
df.head()

#2提取特征变量和目标变量
X = df.drop(columns='响应')
y = df['响应']

#3划分训练集与测试集合
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

#4模型训练
from sklearn.ensemble import AdaBoostClassifier
clf = AdaBoostClassifier(random_state=123)
clf.fit(X_train, y_train)

#5模型预测及评估
y_pred = clf.predict(X_test)   #对测试集数据进行预测
print(y_pred)
a = pd.DataFrame()
a['预测值'] = list(y_pred)
a['实际值'] = list(y_test)
print(a.head())

#所有测试集数据的预测准确度
from sklearn.metrics import accuracy_score
score = accuracy_score(y_pred, y_test)
print('预测准确度：' + str(score))

#预测属于各个分类的概率（AdaBoost分类模型的弱学习器分类决策树模型在本质上预测的并不是准确的0或1的分类而是预测属于某一分类的概率）。
y_pred_proba = clf.predict_proba(X_test)
#print('预测各个分类的概率：' + str(y_pred_proba))

#6绘制ROC曲线来评估模型的预测效果
from sklearn.metrics import roc_curve
fpr, tpr, thres = roc_curve(y_test.values, y_pred_proba[:,1])
import matplotlib.pylab as plt
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS'] #用来正常显示中文
plt.plot(fpr, tpr)
plt.title('ROC')
plt.xlabel('假警报率')
plt.ylabel('命中率')
plt.show()

#计算模型的ACU值
from sklearn.metrics import roc_auc_score
score = roc_auc_score(y_test, y_pred_proba[:,1])
print('AUC:'+ str(score))      #ACU值为0.904

#分析数据特征的重要性，并对特征名称和特征重要性进行汇总
features = X.columns
importances = clf.feature_importances_
#整理成二维表格，并按特征重要性降序排列
importances_df = pd.DataFrame()
importances_df['特征名称'] = features
importances_df['特征重要性'] = importances
importances_df.sort_values('特征重要性', ascending=False)
print(importances_df.sort_values('特征重要性', ascending=False))
#结果解读：特征重要性最高的特征变量是"月消费"，其次是"月消费/月收入"和"月收入"

#7参数调优
import numpy as np
from sklearn.model_selection import GridSearchCV
parameters = {'n_estimators':np.arange(5, 25, 5)}      #'n_estimators'为弱学习器(决策树的个数）默认为10个(int型）
new_model = AdaBoostClassifier(random_state=1)   #构建新模型
grid_search = GridSearchCV(new_model, parameters, cv=6, scoring='accuracy')
#cv参数设置为6表示交叉验证6次，设置模型评估标准scoring参数为'accuracy'，即以准确度作为评估标准，也可以设置成'roc_auc'则表示以ROC曲线的AUC值作为评估标准
grid_search.fit(X_train, y_train)
c = grid_search.best_params_
print(c)   #参数调优结果：{  'n_estimators': 20}  再重新更换模型参数，结果更优


#AdaBoostClassifier(分类决策树模型)参数如下
#参数：base_estimator    含义：弱学习器类型     取值：一般选择决策树模型或MLP神经网络模型，默认为分类决策树模型
#参数：n_estimators      含义：弱学习器的最大迭代次数   取值：int型数据，默认值取50，如果训练集已经完美地训练好，算法可能会提前停止
#参数：learning_rate     含义：弱学习器的权重缩减系数   取值：取之范围为(0,1],取值较小意味着达到一定的误分类数或学习效果需要更多迭代次数和更多弱学习器，默认取值0.1,即不缩减
#参数：algorithm         含义：算法         取值：取值为'SAMME'，代表使用对样本集分类的效果调整弱学习器权重，取值为'SAMME.R'代表使用对样本集分类的预测概率调整弱学习器权重。默认取'SAMME.R'
#参数：random_state      含义：设置随机状态  取值：取值范围为{int型数据，randomstate实例，None}。如果为int型数据，则指定了随机数生成器的种子，每次运行的结果都是一致的，如果为randomstate实例则指定了随机数生成器；如果为None，则使用默认的随机数生成器，默认取none


#AdaBoostRegressor(回归决策树模型)参数如下
#参数：base_estimator    含义：弱学习器类型     取值：一般选择决策树模型或MLP神经网络模型，默认为分类决策树模型
#参数：n_estimators      含义：弱学习器的最大迭代次数   取值：int型数据，默认值取50，如果训练集已经完美地训练好，算法可能会提前停止
#参数：learning_rate     含义：弱学习器的权重缩减系数   取值：取之范围为(0,1),取值较小意味着达到一定的误分类数或学习效果需要更多迭代次数和更多弱学习器，默认取值1.0,即不缩减
#参数：algorithm         含义：算法         取值：取值为'SAMME'，代表使用对样本集分类的效果调整弱学习器权重，取值为'SAMME.R'代表使用对样本集分类的预测概率调整弱学习器权重。默认取'SAMME.R'
#参数：random_state      含义：设置随机状态  取值：取值范围为{int型数据，randomstate实例，None}。如果为int型数据，则指定了随机数生成器的种子，每次运行的结果都是一致的，如果为randomstate实例则指定了随机数生成器；如果为None，则使用默认的随机数生成器，默认取none
#参数：loss              含义：更新权重的损失函数    取值：取值范围为{'linear', 'square', 'exponen-tial'}，其中'linear'为线性损失函数，'square'为平方损失函数，'exponential'为指数损失函数，默认取'linear'线性损失函数

#  模型参数（选学）可在Jupyter notebook中输入并运行以下代码来查看官方说明文档内容
#from sklearn.ensemble import AdaBoostClassifier
#AdaBoostClassifier?

#from sklearn.ensemble import AdaBoostRegressor
#AdaBoostRegressor?