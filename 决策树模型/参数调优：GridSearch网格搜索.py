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
print('整体预测准确度：' + str(score))  #打印结果0.896
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

#计算模型的预测准确度
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

#6参数调优：k折交叉验证
from sklearn.model_selection import cross_val_score  #引入交叉验证的函数cross_val_score（）
acc = cross_val_score(model, x, y, cv=5)      #用cross_val_score() 函数进行交叉验证，传入的参数依次为模型名称（model）、特征变量（x）、目标变量（y）、交叉验证的次数（cv）
#这里设置从cv为5，表示交叉验证5次，每次随机取4/5的数据用于训练，1/5的数据用于测试，如果不设置参数，则默认设置参数为3次。
#此外，这里没有设置scoring参数，表示以默认值'accuracy'（准确度）作为评估标准
#如果想以roc曲线的AUC的值作为评估标准，则可以设置scoring参数为'roc_auc'（acc = cross_val_score(model, x, y, scoring='roc_auc', cv=5)
print(acc)  #查看5次交叉验证得到的打分
print(acc.mean())  #获得上面5个得分的平均分

#7GridSearch网格搜索
#GridSearch网格搜索是一种穷举搜索的参数调优手段：遍历所有候选参数，循环建立模型并评估模型的有效性和准确性，选取表现最好的参数作为最终结果。
#以决策树模型的最大深度参数max_depth为例，我们可以在[1,3,5,7,9]这些值中遍历，以准确度或roc曲线的auc值作为评估标准来搜索最合适的max_depth值
#如果要同时调节多个模型参数，例如，模型有2个参数，第一个参数有4种可能，第二种参数有5中可能，所有的可能性可以表示4*5的网格，那么遍历的过程就像是在网格（Grid）里搜索（Search），这就是该方法名称的由来。

#1单参数调优
#这里以但参数max_depth为例，演示机器学习中如何用网格搜索进行参数调优
from sklearn.model_selection import GridSearchCV   #从Scikit-Learn库中引入GridSearchCV（）函数
parameters = {'max_depth': [ 1,3, 5, 7, 9]}        #指定决策树模型中待调优参数max_depth的候选值范围
#parameters = {'max_depth': np.arange(1, 10, 2)}   #等同上面，批量生成候选值范围，左开右闭，间隔为2
model = DecisionTreeClassifier()                   #构建决策树模型并将其赋给变量model
grid_search = GridSearchCV(model, parameters, scoring='roc_auc', cv=5)   #将决策树模型和待调优参数的候选值范围传入GridSearchCV（）函数
#并设置参数scoring参数为'roc_acu',表示以roc曲线的ACU值作为评估标准，如果不设置则以默认值'accuracy'（准确度）作为评估标准，设置cv参数为5，表示进行5折交叉验证

#将数据传入网格搜索模型并输出参数的最优值
grid_search.fit(x_train, y_train)  #传入训练集数据并开始进行参数调优
grid_search.best_params_    #输出参数的最优值
print(grid_search.best_params_ )  #打印输出
#输出解说
#因为max_depth参数设置来5个候选值，又设置来5折交叉验证，所以对于每一个候选值，模型都会运行5遍（共运行5*5=25遍），每个候选值都通过5折交叉验证获得一个平均分
#根据平均分进行排序，得到参数的最优值就如打印输出所示，即决策树的最大深度设置为5时最优（{'max_depth': 5}）

#使用获得的参数最优值重新搭建模型，并通过查看新模型的预测准确度及roc曲线的AUC值来验证参数调优后是否提高来模型的有效性
model = DecisionTreeClassifier(max_depth=5)
model.fit(x_train, y_train)
#把测试集数据导入模型进行预测，并用accuracy_score（）函数查看整体预测准确度
y_pred = model.predict(x_test)
from sklearn.metrics import accuracy_score  #accuracy_score（）函数查看整体预测准确度
score = accuracy_score(y_pred, y_test)   #传入测试集目标变量和实际目标变量
print('调优后模型预测准确度：' + str(score))
#打印结果解读
#得到新模型在测试集上的预测准确度为0.970，即3000个测试集数据中有2910人的预测结果与实际结果相符，与原模型在测试集上的预测准确度0.896相比，参数调优之后预测准确度有所上升
#其实预测准确度也有可能下降，因为参数调优时是以roc曲线的AUC值（scoring='roc_auc')作为评估标准的，而非预测准确度

#查看完预测准确度，接着来查看roc曲线的auc值
y_pred_proba = model.predict_proba(x_test)  #查看预测属于各个分类的概率
from sklearn.metrics import roc_auc_score
score = roc_auc_score(y_test.values, y_pred_proba[:,1])
print(score)  #打印结果0.972
#获得的acu值为0.972，与参数调优前的acu值0.945相比，模型的有效性的确有所提高

#补充知识点：决策树深度增加时特征重要性的改变(参数调优后，树的子节点和叶子节点都会有所增加，特征重要性也可能发生变化）
print(model.feature_importances_)   #特征变量的数值[0.         0.51714576 0.10142678 0.05863533 0.08861616 0.23417597]
#对比原模型的特征变量的特征重要性发现除工资没发生变化，增加决策树深度其他特征变量都发生来变化


#1.2多参数调优
from sklearn.model_selection import GridSearchCV
#指定决策树模型格参数的候选值范围
parameters = {'max_depth':[5, 7, 9, 11, 13], 'criterion':['gini', 'entropy'], 'min_samples_split':[5, 7, 9, 11, 13, 15]}
#criterion:特征选择标准，取值为'entropy'（信息熵）和'gini'（基尼系数），默认值为'gini'
#max_depth:决策树最大深度，取值为int型数据或None，默认值为None。一般数据或特征较少时可以不设置，如果数据或特征较多，可以设置最大深度进行限制
#min_samples_split：子节点往下分裂所需的最小样本数，默认值为2。如果子节点中的样本数小于该值则停止分裂


#构建决策树模型
model = DecisionTreeClassifier()
#网格搜索
grid_search = GridSearchCV(model, parameters, scoring='roc_auc', cv=5)
grid_search.fit(x_train, y_train)
#输出参数的最优值
print(grid_search.best_params_)#输出结果{'criterion': 'gini', 'max_depth': 9, 'min_samples_split': 15}
#从输出结果可知将criterion设置为'gini'（基尼系数）、max_depth设置为9、min_samples_split设置为15时，模型最优。
#再将这些参数的最优值引入模型，代码如下
#model = DecisionTreeClassifier(criterion='gini', max_depth=9, min_samples_split=15)


#使用获得的参数最优值重新搭建模型，并通过查看新模型的预测准确度及roc曲线的AUC值来验证参数调优后是否提高来模型的有效性
model = DecisionTreeClassifier(criterion='gini', max_depth=9, min_samples_split=15)
model.fit(x_train, y_train)
#把测试集数据导入模型进行预测，并用accuracy_score（）函数查看整体预测准确度
y_pred = model.predict(x_test)
from sklearn.metrics import accuracy_score  #accuracy_score（）函数查看整体预测准确度
score = accuracy_score(y_pred, y_test)   #传入测试集目标变量和实际目标变量
print('多参数调优后模型预测准确度：' + str(score))
#打印结果解读
#得到新模型在测试集上的预测准确度为0.975比单参数调优的0.970好一点，即3000个测试集数据中有2910人的预测结果与实际结果相符，与原模型在测试集上的预测准确度0.896相比，参数调优之后预测准确度有所上升


#查看完预测准确度，接着来查看roc曲线的auc值
y_pred_proba = model.predict_proba(x_test)  #查看预测属于各个分类的概率
from sklearn.metrics import roc_auc_score
score = roc_auc_score(y_test.values, y_pred_proba[:,1])
print(score)  #打印结果0.969
#获得的acu值为0.968，与单参数调优前的acu值0.972相比，模型的有效性的确有所下降一点，但其准确度高了一点

#注意事项
#第一多参数调优和单参数分别调优是有区别的。不能为了省事，对多个参数分别进行单参数调优，然后将结果汇总，这种做法是不严谨的。
#因为在进行单参数调优时，其他参数会取默认值，那么就忽略了该参数和其他参数都不取默认值的情况，即忽略了多个参数对模型对组合影响。
#以上述代码示例来说，进行多参数调优时，有5*2*6=60种可能的组合，而进行3次单参数调优时，则只有5+2+6=13种可能的组合。因此，如果只需要调节一个参数，那么可以进行单参数调优，如果需要调节多个参数，则推荐多参数调优

#第二，如果使用GridSearchCV（）函数得到的参数最优值是给定范围的边界值，那么有可能存在范围以外的值使得模型效果更好，此时需要额外增大范围，继续进行参数调优
#举例来说，倘若上述代码获得的max_depth最优值为设定的最大值13，那么真正的max_depth最优值可能更大，此时便需要重新调整搜索网络，如将max_depth的搜索范围变成【9，11，13，15，17】，再重新进行参数调优


#补充知识点：决策树的前剪枝和后剪枝
#决策树剪枝的目的是防止构建的决策树出现过拟合。决策树剪枝分为前剪枝和后剪枝，两者的定义如下
#前剪枝：从上往下剪枝，通常利用超参数进行剪枝。例如，通过限制树的最大深度（max_depth）便能剪去该最大深度下面的节点
#后剪枝：从下往上剪枝，大多是根据业务需求剪枝。例如，在违约预测模型中，认为违约概率为45%和50%的俩个叶子节点都是高危人群，那么就把这俩个叶子节点合并成一个节点
#在商业实战中，前剪枝应用更广泛，参数调优其实也起到了一定的剪枝作用

#补充知识点： 分类决策树模型DecisionTreeClassifier()的常用超参数
#criterion:特征选择标准，取值为'entropy'（信息熵）和'gini'（基尼系数），默认值为'gini'
#splitter：取值为'best'和'random'。'best'指在特征的所有划分点中找出最优的划分点，适合样本量不大的情况；'random'指随机地在部分划分点中寻找局部最优的划分点，适合样本量非常大的情况；默认值为'best'
#max_depth:决策树最大深度，取值为int型数据或None，默认值为None。一般数据或特征较少时可以不设置，如果数据或特征较多，可以设置最大深度进行限制
#min_samples_split：子节点往下分裂所需的最小样本数，默认值为2。如果子节点中的样本数小于该值则停止分裂

#min_samples_leaf:叶子节点最小样本数，默认值为1。如果叶子节点中的样本数小于该值，该叶子节点会和兄弟节点一起被剪枝，即剔除该叶子节点和其兄弟节点，并停止分裂
#min_weight_fraction_leaf:叶子节点最小的样本权重和，默认值为0，即不考虑权重问题。如果小于该值，该叶子节点会和兄弟节点一起剪枝。如果较多样本有缺失值或者样本的分布类别偏差很大，则需考虑样本权重问题
#max_features:在划分节点时所考虑的特征值数量的最大值，默认值为None，可以传入int型或float型数据，如果传入的是float型数据，则表示百分数
#max_leaf_nodes:最大叶子节点数，默认值为None，可以传入int型数据
#class_weight:指定类别权重，默认值为None，可以取'balanced'，代表样本量少的类别所对应的样本权重过高，也可以传入字典来指定权重。
# 该参数主要是为防止训练集中某些类别的样本过多，导致训练的决策树过于偏向这些类别。除来指定该参数，还可以使用过采样和欠采样的方法处理样本类别不平衡的问题
#random_state:当数据量较大或特征变量较多，可能在某个节点划分时，会遇到两个特征变量的信息熵增益或基尼系数下降值相同的情况，
# 此时决策树模型默认会从中随机选择一个特征变量进行划分，这样可能会导致每次运行程序后生成的决策树不一致。设置random_stata参数（如设置为123）
#可以保证每次运行程序后各节点的分裂结果都是一致的，这在特征变量较多、树的深度较深时较为重要


#补充知识点：树模型在金融大数据风控领域的应用
#以决策树为代表的树模型在金融大数据风控领域也有很大的应用空间。以银行的信贷违约预测模型为例，通常会用到逻辑回归模型和决策树模型
#逻辑回归模型不需要太多变量，不容易过拟合，泛化能力较强，可能一年才需要换一次模型，但有时不够精确，不能有效剔除潜在违约人员。
#树模型（决策树、随机森林、XGBoost等模型）不太稳定（一个变量可以反复用），容易造成过拟合，泛化能力较弱，一段时间后换一批人可能就不行了，但拟合度强，区分度高，可以快速去掉违约人员。
#因此，商业实战中常以基于逻辑回归的评分卡模型为基础(稳定性强,半年到一年更新一次,但不够精确,ks值不够大),再结合决策树等树模型(不太稳定,可能要一个月更新一次,但拟合度强,区分度高,可以在第一拨快速去掉违约人员

#总结来说作为机器学习但经典算法模型,决策树模型具有独特的优势,如对异常值不敏感、可解释性强等,不过它也有一些缺点,如结果不稳定、容易造成过拟合等
#更重要的是决策树模型是很多重要集成模型的基础.