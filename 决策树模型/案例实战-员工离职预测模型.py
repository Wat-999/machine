#1数据读取和预处理
import pandas as pd
df = pd.read_excel('/Users/macbookair/Desktop/数据分析/书本配套资料及电子书/python/Python大数据分析与机器学习/源代码汇总-2020-12-16/第5章 决策树模型/源代码汇总_PyCharm格式/员工离职预测模型.xlsx')
#print(df.head())  #打印前5行数据
df = df.replace({'工资': {'低':0, '中':1, '高':2}})  #用replace（）函数将文本"高"中"低"分别替换为数字2，1，0
print(df.head())

#2提取特征变量和目标变量
x = df.drop(columns='离职')   #用drop函数删除'离职'列，将剩下的数据作为特征变量赋给变量x
y = df['离职']       #用DataFrame提取列的方式提取'离职'列作为目标变量，并赋给变量y

#3划分训练集与测试集
from sklearn.model_selection import train_test_split   #从sklearn-Learn库中引入train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.2, random_state=123 )

#4模型训练及搭建
from sklearn.tree import DecisionTreeClassifier    #从sklearn-Learn库中引入分类决策树模型DecisionTreeClassifier
model = DecisionTreeClassifier(max_depth=3, random_state=123)   ##引入模型设置决策树最大深度参数max_depth=3（即根节点下分3层），随机状态参数random_state=0为0，这里0没有意义可以换成其他数字，它是一个种子参数，可使每次运行结果一致
model.fit(x_train, y_train)       #传入前面划分出来的训练集数据

#5模型预测及评估
#1直接预测是否离职
y_pred = model.predict(x_test)   #传入测试集数据
print(y_pred)
#print(y_pred[0:100])     #打印前100行预测结果
a = pd.DataFrame()        #创建一个空DataFrame
a['预测值'] = list(y_pred)
a['实际值'] = list(y_test)
print(a.head())

#查看整体的预测准确度
from sklearn.metrics import accuracy_score
score = accuracy_score(y_pred, y_test)
print('整体预测准确度：' + str(score))
#print(model.score(x_test, y_test))   #等同上面的结果

#2预测不离职和离职的概率
#其实分类决策树模型在本质上预测的并不是精确的0或1的分类，而是预测属于某一分类的概率
y_pred_proba = model.predict_proba(x_test)    #传入测试集数据
b = pd.DataFrame(y_pred_proba, columns=['不离职概率', '离职概率'])  #获得的y_pred_proba是一个二维数组，用columns修改列名
print(b.head())

#3模型预测效果评估
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
print('模型预测准确度：' + str(score))  #获得的ACU值为0.945，预测效果还是挺不错的

#4特征重要性评估
#模型搭建完成后，有时还需要知道各个特征变量的重要程度，即哪些特征变量在模型中发挥的作用更大，这个重要程度称为特征重要性。
#在决策树模型中，一个特征变量对模型整体的基尼系数下降的贡献越大，它的特征重要性就越大
#print(model.feature_importances_)   #打印决策树中各特征变量的特征重要性，注意这些特征的重要性之和为1

#如果特征变量很多，使用如下代码将特征名称和特征重要性一一对应，以方便查看
features = x.columns     #取特征名称
importances = model.feature_importances_   #取特征重要性
#以二维表格形式显示
importances_df = pd.DataFrame()   #构造空的DataFrame()
importances_df['特征名称'] = features
importances_df['特征重要性'] = importances
importances_df.sort_values('特征重要性', ascending=False)   #用sort_values（）函数将表格特征按特征重要性进行降序排序
print(importances_df.sort_values('特征重要性', ascending=False))  #输出二维表格，可以看到特征重要性最高的'满意度'，这一点也符合常理

#表格解说
#可以看到特征重要性最高的'满意度'，这一点也符合常理，因为员工对工作的满意度高，其离职的概率就相对较低，反之则高。
#其次重要的是'考核得分'和'工龄'。'工资'在该模型中的特征重要性为0，也就是说它没有发挥作用，这并不符合常理。
#之所以会有这个结果，在某种程度上是因为我们限制了决策树的最大深度为3层（max_depth=3），所以'工资'没有发挥作用的机会，如果增大决策树的最大深度，那么它可能发挥作用。
#另外一个更具体的原因是本案例中的'工资'不是具体的数值，而是'高''中''低'三个档次，这种划分过于宽泛，使得该特征变量在决策树模型中发挥的作用较小，如果'工资'是具体的数值，如1000元，那么该特征变量应该会发更大的作用





