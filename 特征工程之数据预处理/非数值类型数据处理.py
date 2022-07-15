#非数值型数据处理
#机器学习建模时处理的都是数值类型的数据，然而实际工作中获取的数据往往会包含非数值类型的数据，其中最常见的是文本类型数据，
#例如，性别中的男和女，处理时可以用查找、替换的思路，分别转换成数字1和0。但如果类别有很多，又该如何处理了
#这时就用Get_dummies哑变量处理和Label Enconding编号处理

#哑变量也叫虚拟变量，通常取值为0或1，通常利用pandas库中的get_dummies()函数进行哑变量处理，它不仅可以处理只有两个分类的简单问题，
#还可以处理含有多个分类的问题

import pandas as pd
df = pd.DataFrame({'客户编号': [1, 2, 3], '性别': ['男', '女', '男']})
print(df)

#接着用get_dummies()函数对文本类型的数据进行处理，第一个参数为表格名称，第二个参数为需要处理的列名
df = pd.get_dummies(df, columns=['性别'])
print(df)
#可以看到原来的性别列变成性别_女和性别_男两列，这两列中的数字1表示符合列名，数字0表示不符合列名
#虽然现在已经将文本类型的数据已经转换成列数字，但是性别_女和性别_男两列存在多重共线性，即知道其中一列内容，就能知道另一列的内容，即性别_女=1-性别_男
#多重共线性会带来一系列问题，因此需要用drop（）函数删除一列
df = df.drop(columns='性别_女')
df = df.rename(columns={'性别_男':'性别'})    #用rename()函数更改列名
print(df);
#至此便完成了非数值类型数据的哑变量处理

#2房屋朝向的数值转换
import pandas as pd
df = pd.DataFrame({'房屋编号': [1, 2, 3, 4, 5], '朝向': ['东', '南', '西', '北', '南']})
print(df)

#构造哑变量
df = pd.get_dummies(df, columns=['朝向'])
print(df)

#上表同样存在多重共线性（即根据3个朝向的数字就能判断第4个朝向的数字是0还是1），因此需要从新构造出来的4个哑变量中删除一个
df = df.drop(columns='朝向_西')
print(df)
#这样便通过哑变量处理将分类变量转换为数值变量，为后续构建模型打好了基础。
#构造哑变量容易产生高维数据，因此，哑变量常和PCA(主成分)一起使用，即构造哑变量产生高维数据后用PCA进行降维

#Label Encoding编号处理
#除了使用get_dummies()函数进行非数值类型处理外，还可以使用Label Encoding进行编号处理，具体来说是使用LabelEncoder()函数将文本类型的数据转换成数字
import pandas as pd
df = pd.DataFrame({'编号': [1, 2, 3, 4, 5], '城市': ['北京', '上海', '广州', '深圳', '北京']})
print(df)

#将城市列的文本内容转换为不同数字
from sklearn.preprocessing import LabelEncoder  #引入LabelEncoder()函数
le = LabelEncoder()               #赋给变量le
label = le.fit_transform(df['城市'])   #用fit_transform()函数将待转化的列传入模型
print(label)  #可以看到北京被转化成数字1，上海被转化成数字0，等
df['城市'] = label   #用转换结果替换原来的列聂荣
print(df)
#上述示例中使用Label Encoding处理后产生了一个奇怪的现象：上海和广州的平均值是北京，这个现象其实是没有意义的，这也是Label Encoding的一个缺点
#可能产生一些没意义的关系。不过树模型（如决策树、随机森林及XGBoost集成算法）能很好地处理这种转化，因此对于树模型来说，这种奇怪的现象是不会影响结果的。

#补充知识点pandas库中的raplace()函数
#Label Encoding（）函数生成的数字是随机的，如果想按照特定内容进行替换，可以使用repleace（）函数，这两种处理方式对于建模效果不会有太大影响
df = pd.DataFrame({'编号': [1, 2, 3, 4, 5], '城市': ['北京', '上海', '广州', '深圳', '北京']})
#在使用replace（）函数之前，先利用value_counts()函数查看"城市"列有哪些内容需要替换（因为有时数据量很大，通过人眼判断可能会遗漏某些内容）
print(df['城市'].value_counts())   #value_counts()可以统计非重复项出现的次数

#从上述结果可知，需要替换的是'北京'上海'深圳'广州'这4个词，这里用replace（）函数按'北上广深'的顺序进行数字编号
df['城市'] = df['城市'].replace({'北京': 0, '上海': 1, '广州': 2, '深圳': 3})
print(df)

#可以看到Label Encoding（）函数是对文本内容进行随机编号，而用replace（）函数可以将文本内容替换成自定义的值。
#不过当分类较多时，还需要先用value.counts（）函数获取每个分类的名称，步骤会稍微烦琐一些
#总计来说，Get_dummies的优点是它的值只有0和1，缺点是当类别的数量很多时间，特征维度会很高，此时可以配合使用PCA（主成分分析）来减少维度
#如果类别数量不多，可以优先考虑使用Get_dummies,其次考虑使用Label Encoding或replace()函数，但是如果基于树模型的机器学习模型，用Label Encoding也没有太大关系






