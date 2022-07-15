#数据读取与预处理
#利用apyori库和mlxtend库来编写代码前，需要读取数据并进行简单的预处理，因为两个库可处理的都是如下所示的双重列表结构(transactions)，
#而用pandas库读取Excel工作薄得到的是Dataframe格式的数据结构，所以需要进行转换。
#transactions = [['A', 'B', 'C'], ['A', 'B'], ['B', 'C'], ['A', 'B', 'C', 'D'], ['B', 'C', 'D']]

#1数据读取
import pandas as pd
df = pd.read_excel('/Users/macbookair/Desktop/数据分析/书本配套资料及电子书/python/Python大数据分析与机器学习/源代码汇总-2020-12-16/第15章 关联规则分析-Apriori模型/源代码汇总_PyCharm格式/中医病症.xlsx')
df.head()

#2数据预处理：需要提取"病人症状"列的内容，并将其装换为双重列表结构
symptoms = []    #创建一个空列表symptoms(病症），用来存储之后提取的每一个患者的病症数据
for i in df['病人症状'].tolist():        #遍历每个患者的"患者病症"列，用用tolist()函数将该列的内容装换成一个列表，列表中的每个元素就是每个患者的所有病症
    symptoms.append(i.split(','))       #先用split()函数按逗号(,)对列表元素进行分割，将患者的一个个病症分割开来，并存储在一个个子列表中，再用append()函数将所有患者的病症子列表汇总到symptoms列表中
    print(symptoms)

#关联规则分析(apyori库）
from apyori import apriori
rules = apriori(symptoms, min_support=0.1, min_confidence=0.7)   #min_support最小支持度(因为本案例的数据量较小，所以设置为较小的值，这样能多发现一些关联规则， min_confidence最小置信度
results = list(rules)   #用list函数将获得的关联关系rules装换为列表，方便后面调用

#提取results中的关联规则，并通过字符串拼接来更好地呈现关联规则
for i in results:        #遍历results中的每一个频繁项集
    for j in i.ordered_statistics:   #获取频繁项集中的关联规则
        X = j.items_base             #关联规则的前件
        Y = j.items_add              #关联规则的后件
        x = ','.join([item for item in X])        #连接前件中的元素
        y = ','.join([item for item in Y])        #连接后件中的元素
        if x != '':                               #防止出现关联规则中前件为空的情况
            print(x + '→' + y)     #通过字符串拼接
#结果解读，可以看到，在获得的关联规则中，的确有之前提到的同一脏器导致的病症关联规则，如便秘和消化不良(脾的关联病症）的关联规则
#并且还有不同脏器之间相互影响导致的病症关联规则，如脱发(肾的关联病症）和眼干(肝的关联病症）的关联规则，其余的关联规则也说明了不同病症之间存在一些关联性

#方法二关联规则分析(mlxtend库）
from mlxtend.preprocessing import TransactionEncoder
TE = TransactionEncoder()
data = TE.fit_transform(symptoms)
print(data)

import pandas as pd
df = pd.DataFrame(data, columns=TE.columns_)
df.head()

from mlxtend.frequent_patterns import apriori
items = apriori(df, min_support=0.1, use_colnames=True)
print(items)

print(items[items['itemsets'].apply(lambda x: len(x)) >= 2])


from mlxtend.frequent_patterns import association_rules
rules = association_rules(items, min_threshold=0.7)
print(rules)

for i, j in rules.iterrows():
    X = j['antecedents']
    Y = j['consequents']
    x = ', '.join([item for item in X])
    y = ', '.join([item for item in Y])
    print(x + ' → ' + y)


