# AdaBoost信用卡精准营销模型

# 1.读取数据
import pandas as pd
df = pd.read_excel('/Users/macbookair/Desktop/数据分析/书本配套资料及电子书/python/Python大数据分析与机器学习/源代码汇总-2020-12-16/第9章 AdaBoost与GBDT模型/源代码汇总_Pycharm/信用卡精准营销模型.xlsx')
df.head()

# 2.提取特征变量和目标变量
X = df.drop(columns='响应')
y = df['响应']

# 3.划分训练集和测试集
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

# 4.模型训练及搭建
from sklearn.ensemble import AdaBoostClassifier
clf = AdaBoostClassifier(random_state=123)
clf.fit(X_train, y_train)

# #5 模型预测及评估
y_pred = clf.predict(X_test)
print(y_pred)

a = pd.DataFrame()
a['预测值'] = list(y_pred)
a['实际值'] = list(y_test)
a.head()

#所有测试集数据的预测准确度
from sklearn.metrics import accuracy_score
score = accuracy_score(y_pred, y_test)
print('预测准确度：' + str(score))

#预测属于各个分类的概率
y_pred_proba = clf.predict_proba(X_test)
print(y_pred_proba[0:5])

#绘制ROC曲线来评估模型的预测效果
from sklearn.metrics import roc_curve
fpr, tpr, thres = roc_curve(y_test.values, y_pred_proba[:,1])
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




