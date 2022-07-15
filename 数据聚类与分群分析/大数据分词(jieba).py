# # 补充知识点1：大数据分词
# 1.jieba库的安装与简单演示
import jieba
word = jieba.cut('我爱北京天安门')
for i in word:
    print(i)

# 2.读取文本内容，并进行分词
import jieba
report = open("/Users/macbookair/Desktop/数据分析/书本配套资料及电子书/python/Python大数据分析与机器学习/源代码汇总-2020-12-16/第13章 数据聚类与分群/源代码汇总_PyCharm格式/信托行业年度报告.txt", "r", encoding="gbk").read()
#注意报无法打开文件时，就是encoding参数设置有问题，一般设置encoding="utf-8"，这里尝试错误改成来encoding="gbk"
words = jieba.cut(report)

for word in words:
    print(word)

# 3.提取分词后的4字词
words = jieba.cut(report)  # 这里得重新jieba.cut()一下，因为之前的words用过一次就被清空了
report_words = []
for word in words:  # 将大于4个字的词语放入列表
    if len(word) >= 4:
        report_words.append(word)
print(report_words)

# 4.统计高频词汇的词频
from collections import Counter
result = Counter(report_words)
print(result)

result = Counter(report_words).most_common(50)  # 取最多的50组
print(result)

import jieba
from collections import Counter

# 1.读取文本内容，并利用jieba.cut功能俩进行自动分词
report = open("/Users/macbookair/Desktop/数据分析/书本配套资料及电子书/python/Python大数据分析与机器学习/源代码汇总-2020-12-16/第13章 数据聚类与分群/源代码汇总_PyCharm格式/信托行业年度报告.txt", "r", encoding="gbk").read()
words = jieba.cut(report)

# 2.通过for循环来提取words列表中大于4个字的词语
report_words = []
for word in words:  # 将大于4个字的词语放入列表
    if len(word) >= 4:
        report_words.append(word)
print(report_words)

# 3.获得打印输出高频词的出现次数
result = Counter(report_words).most_common(50)  # 取词频最高的50组词
print(result)


