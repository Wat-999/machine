#1读取数据
import pandas as pd
df = pd.read_excel('/Users/macbookair/Desktop/数据分析/书本配套资料及电子书/python/Python大数据分析与机器学习/源代码汇总-2020-12-16/第6章 朴素贝叶斯模型/源代码汇总_PyCharm格式/肿瘤数据.xlsx')
print(df.head())

#2划分特征变量与目标变量
X = df.drop(columns='肿瘤性质')   #用drop（）函数删除'肿瘤性质'列，将剩下的数据作为特征变量赋给变量X
y = df['肿瘤性质']    #通过dataframe提取列的方式提取'肿瘤性质'列的数据作为目标变量，并赋给变量y

#3模型搭建与使用
#1划分训练集与测试集
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2, random_state=1)

#2模型搭建
from sklearn.naive_bayes import GaussianNB
nb_clf = GaussianNB()  #高斯朴素贝叶斯模型
nb_clf.fit(X_train, y_train)

#3模型预测与评估
y_pred = nb_clf.predict(X_test)
print(y_pred[:100])
a = pd.DataFrame()
a['预测值'] = list(y_pred)
a['实际值'] = list(y_test)
print(a.head())  #汇总预测结果

from sklearn.metrics import accuracy_score
score = accuracy_score(y_pred, y_test)
print(score)    #打印整体预测准确度
#预测概率
y_pred_proba = nb_clf.predict_proba(X_test)
print(y_pred_proba[0:5])


#朴素贝叶斯模型属于分类模型，所以也可以利用roc曲线来评估其预测效果，其评估方法与逻辑回归模型和决策树模型的评估方法是一样的。
#要说明的是，这里的6个特征变量是笔者筛选出的特征重要性较高的变量，所以该模型的ROC曲线会比较陡峭，且ACU值会比较高
#总结来说，朴素贝叶斯模型是一种非常经典的机器学习模型，它主要基于贝叶斯公式，在应用过程中会把数据集中特征看成是相互独立的，
#而不需要考虑特征间的关联关系，因此运算速度较快。相比与其他经典的机器学习模型，朴素贝叶斯模型的泛化能力稍弱，
#不过当样本及特征的数量增加时，其他预测效果也是不错的


#4模型预测效果评估
from sklearn.metrics import roc_curve
fpr, tpr, thres = roc_curve(y_test, y_pred_proba[:,1])  #roc_curve()函数传入测试集的目标变量y_test及预测的离职概率y_pred_proba[:,1]，计算出不同阀值下的命中率和假警报率
a = pd.DataFrame()
a['阀值'] = list(thres)
a['假警报率'] = list(fpr)
a['命中率'] = list(tpr)
print(a)

#绘制roc曲线
import matplotlib.pyplot as plt
plt.plot(fpr, tpr)
plt.title('roc曲线')
plt.xlabel('假警报率')
plt.ylabel('命中率')
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS'] #用来正常显示中文
plt.show()

#计算模型的预测准确度（auc）
from sklearn.metrics import roc_auc_score    #引入roc_auc_score（）函数来计算auc
score = roc_auc_score(y_test, y_pred_proba[:,1])   #roc_auc_score()函数传入测试集的目标变量y_test及预测的离职概率y_pred_proba[:,1]，
print('模型预测准确度：' + str(score))  #获得的ACU值为0.979，预测效果还是挺不错的

