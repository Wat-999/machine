transactions = [['A', 'B', 'C'], ['A', 'B'], ['B', 'C'], ['A', 'B', 'C', 'D'], ['B', 'C', 'D']]

from apyori import apriori   #引入apyori库中apriori函数
rules = apriori(transactions, min_support=0.4, min_confidence=0.8)
#min_support=0.4参数为最小支持度， min_confidence=0.8参数为最小置信度
results = list(rules)   #用list函数将获得的关联规则(频繁集）转换为列表，方便之后调用

#提取results中的关联规则，并通过字符串拼接来更好地呈现关联规则
for i in results:        #遍历results中的每一个频繁项集
    for j in i.ordered_statistics:   #获取频繁项集中的关联规则
        X = j.items_base             #关联规则的前件
        Y = j.items_add              #关联规则的后件
        x = ','.join([item for item in X])        #连接前件中的元素
        y = ','.join([item for item in Y])        #连接后件中的元素
        if x != '':                               #防止出现关联规则中前件为空的情况
            print(x + '→' + y)     #通过字符串拼接

#通过mlxtend库实现Apriori算法(使用mlxtend库中的函数可以快速、完整地挖掘出数据中的强关联规则）
transactions = [['A', 'B', 'C'], ['A', 'B'], ['B', 'C'], ['A', 'B', 'C', 'D'], ['B', 'C', 'D']]
#因为mlxtend库中apriori函数可以接受的数据类型为布尔值(又称bool型数据，内容为true、flase）或0和1构成的dataframe，
#所以需要先使用mlxtend库中TransactionEncoder函数对数据进行预处理
from mlxtend.preprocessing import TransactionEncoder
TE = TransactionEncoder()   #构造转换模型
data = TE.fit_transform(transactions)   #将原始数据转换为布尔值

#利用pandas库将转化好的布尔值以dataframe形式存储
import pandas as pd
df = pd.DataFrame(data, columns=TE.columns_)   #存储布尔值

#将数据处理为mxltend库可接受的特定格式后，再从mlxtend库的frequent_patterns模块中阴日apriori函数来挖掘购物篮事务库中的频繁项集
from mlxtend.frequent_patterns import apriori
items = apriori(df, min_support=0.4, use_colnames=True)
#min_support=0.4参数为代表最小支持度为0.4,设置参数use_colnames=True，代表使用变量df的列名作为返回频繁项集中项的名称，
#最后将挖掘出的频繁集赋给变量items，此时items为所有符合最小支持度要求的频繁项集。
print(items)

#查看长度大于等于2的频繁项集，其中主要利用的是pandas库中的apply函数，其作用于itemsets列上获取该列每一个项集的长度，即元素个数
#然后判断长度是否大于等于2并进行筛选
print(items[items['itemsets'].apply(lambda x: len(x)) >= 2])

#根据最小置信度在频繁集中挖掘强关联规则
from mlxtend.frequent_patterns import association_rules
rules = association_rules(items, min_threshold=0.8)#参数min_threshold=0.8即为最小置信度设置为0.8
print(rules)
#结果解读
#以第2条关联规则{A}→{B}为例讲解各例的含义。antecedents列代表关联规则中的前件，如关联规则{A}→{B}中的{A};
#consequents列代表关联规则中的后件，如关联规则{A}→{B}中的{B};
#antecedent support列代表前件的支持度，例如A共出现3次(共5笔事务），所以关联前件的支持度为3/5=0.6
#consequents support列代表后件的支持度，例如B共出现5次(共5笔事务），所以关联后件的支持度为5/5=1
#support列代表关联规则的支持度，例如{A,B}共出现3次，所以关联规则的支持度为3/5=0.6
#confidence列代表该关联规则的置信度，以上述的购物篮事务库为例，关联规则{A}→{B}的置信度计算公式：P({B}|{A})=P({A,B})/P({A})
#项集{A,B,}在所有5笔事务中共出现3次（第1、2、4笔事务），所以P({A,B,})=3/5;项{A}在所有5笔事务中共出现3次(第1、2、4笔事务），置信度为1
#还可以用"关联规则支持度/前件支持度"来计算：conf({A}→{B})=support({A,B})/support(A)=(3/5)/(3/5)=1

#补充知识点
#结果中lift  leverage  conviction这3列都是用来衡量关联度强弱的
#lift列代表该关联规则的提升度，其计算公式为"关联规则支持度/(前件支持度*后件支持度，代入数值0.6/(0.6*1)=1。该值越大，表明X和Y之间关联度越强
#lift({A}→{B})=support({A,B})/support(A)*support(B)
#leverage列代表关联规则的杠杆率，其计算公式为"关联规则支持度-前件支持度*后件支持度，代入数值0.6-0.6*1=0。该值越大，表明X和Y的关联度越强
#leverage({A}→{B})=support({A,B})-support(A)*support(B)
#conviction列代表关联规则的确信度，其计算公式为"(1-后件支持度）/(1-关联规则置信度）。代入数值为(1-1)/(1-1)=∞。该值越大，表明X和Y的关联度越强
#conv({A}→{B})=(1-support({A,B}))/(1-conf({A}→{B}))

#提取results中的关联规则，并通过字符串拼接来更好地呈现关联规则
for i, j in rules.iterrows():        #遍历dataframe中的每一行
    X = j['antecedents']           #关联规则的前件
    Y = j['consequents']            #关联规则的后件
    x = ','.join([item for item in X])        #连接前件中的元素
    y = ','.join([item for item in Y])        #连接后件中的元素
    print(x + '→' + y)     #通过字符串拼接   9条强关联规则都捕捉到了
#其中i表示每一行的序号，j表示每一行的内容，因此j[列名]表示该行对应列名的内容
#x = ','.join([item for item in X])   代表从存储前件的集合中逐一提取前件，并以'，'作为分隔符连接前件中的各个元素
#总体来说apyori库和mlxtend库在实际应用中的差别不大。此外，apyori库在2020年升级后功能更加完善，基本不再遗漏关联规则，建议重点掌握apyori库