#建立模型时，可能会遇到正负样本比例极度不均衡的情况。例如，建立信用违约模型时，违约样本的比例远小于不违约样本的比例
#，此时模型会花更多精力去拟合不违约样本，但实际上找出违约样本更为重要。这会导致模型可能在训练集上表现良好，但测试时表现不佳
#为列改善样本比例不均衡但问题，可以使用过采样和欠采样但方法。假设建立信用违约模型时，样本数据中有1000个不违约样本和100个违约样本。

#过采样的方法有随机采样和SMOTE法采样
#随机过采样是从100个违约样本中随机抽取旧样本作为一个新样本，共反复抽取900次，然后和原来的100个旧样本组合成新的1000个违约样本，和1000个不违约
#样本一起构成新的训练集合。因为随机过采样重复地选取了违约样本，所以有可能造成对违约样本的过拟合

#SMOTE法过采样即合成少数类过采样技术，它是一种针对随机采样容易导致过拟合问题的改进方案。假设对少数类进行4倍过采样，随机选取少数类中一个样本点
#找到离该样本点最近的4个样本点，在选中的样本点和最近的4个样本点分别连成的4条线段上随机随取4点生成新的样本点。之后重复上述步骤，直到少数类的样本个数达到目标为止

#过采样
#1读取数据
import pandas as pd
data = pd.read_excel('/Users/macbookair/Desktop/数据分析/书本配套资料及电子书/python/Python大数据分析与机器学习/源代码汇总-2020-12-16/第11章 特征工程之数据预处理/源代码汇总_PyCharm格式/信用卡数据.xlsx')
data.head()

#2提取特征变量和提取目标变量
X = data.drop(columns='分类')
y = data['分类']

#用collections库中的counter()函数对目标变量进行计数   针对集合中对元素快速计数
from collections import Counter
print(Counter(y))

#为了防止建立信用违约模型，模型着重拟合不违约样本，而无法找出违约样本，采用过采样的方法来改善样本比例不均衡的问题。
#这里分别对随机过采样和SMOTE法过采样进行解决

#1随机过采样(imblearn库专门用来处理数据不均衡问题的工具库)
from imblearn.over_sampling import RandomOverSampler  #引入用来进行随机过采样的RandomOverSampler()函数
ros = RandomOverSampler(random_state=123)      #设置random_state参数保证每次运行结果一致
X_oversampled, y_oversampled = ros.fit_resample(X, y)#使用原始数据的特征变量和目标变量生成过采样数据集，并赋值
print(Counter(y_oversampled))    #用Counter()函数检验一下随机过采样的结果
#结果解读，违约样本数从100上升到1000，与不违约样本数相同，证明随机过采样有效
print(X_oversampled.shape)  #查看X_oversampled特征变量的变化，这里的2000就是违约样本1000和不违约样本1000之和，6即6个特征变量

#2SMOTE法过采样
from imblearn.over_sampling import SMOTE
smote = SMOTE(random_state=123)
X_smotesampled, y_smotesampled = smote.fit_resample(X, y)
print(Counter(y_smotesampled))    #检验SMOTE过采样的效果
#结果解读，违约样本数从100上升到1000，与不违约样本数相同，证明SMOTE过采样有效


#欠采样
#1欠采样是从1000个不违约样本中随机选取100个样本，和100个违约样本一起构成新的训练集。欠采样抛弃了大部分不违约样本
#在搭建模型时有可能产生欠拟合
from imblearn.under_sampling import RandomUnderSampler  #引入用于进行随机欠采样的RandomUnderSampler()函数
rus = RandomUnderSampler(random_state=123)
X_undersampled, y_undersampled = rus.fit_resample(X, y) #使用原始数据的特征变量和目标变量生成过采样数据集，并赋值
print(Counter(y_undersampled))     #用Counter()函数检验一下随机过采样的结果，证明随机欠采样有效
print(Counter(X_undersampled))     #查看X_undersampled特征变量的变化，这里的200就是违约样本100和不违约样本100之和，6即6个特征变量，
#可以看到，随机欠采样后特征变量的数据随之减少

#在实战中处理样本不均衡问题时，如果样本数据量不大，通常使用过采样，因为这样能更好的利用数据，不会像欠采样那样很多数据都没有使用到；
#如果数据量充足，则过采样和欠采样都可以考虑使用

#总体来说，数据预处理是数据分析中非常重要的环节，尽管其相对而言比较枯燥，却能在很大程度上影响模型的效果


