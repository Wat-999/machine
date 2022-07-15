import  pandas as pd
df1 = pd.DataFrame({'公司': ['万科', '阿里', '百度'], '分数': [90.0, 95.0, 85.0]})
df2 = pd.DataFrame({'公司': ['万科', '阿里', '京东'], '股价': [20.0, 180.0, 30.0]})
#df3 = pd.merge(df1, df2)    #使用merge（）函数可以根据一个或或多个键（列）将不同数据表中的行连接起来，
# print(df3)                #在不指定的情况下，默认直接根据相同的列名（公司）对俩个表进行列合并


# df3 = pd.merge(df1, df2, on='公司')    #可以通过on参数指定按照哪一列进行合并，默认的合并是取交集
# print(df3)

# df3 = pd.merge(df1, df2, how='outer')    #可以设置how参数为outer，取并集
# print(df3)

# df3 = pd.merge(df1, df2, how='left')    #可以设置how参数为left，为左连接，保全左表df1的全部内容
# print(df3)                               #如果想保留右表，可以设置how参数为right，为右连接，保全右表df2的全部内容

# df3 = pd.merge(df1, df2, left_index=True, right_index=True)  #如果想根据行索引进行合并，可以设置left_index=True, right_index=True
# print(df3)

# df3 = df1.join(df2, lsuffix='_x', rsuffix='_y')  #注意在使用join函数进行拼接时，若俩个表中存在同名列，则需要设置lsuffix参数（左表同名列的后缀，suffix是后缀的意思，l表示left）
# print(df3)   #和rsuffix参数（右表同名列的后缀，r表示right）；若没有同名列，则可以直接写成df1.join（df2）,这种写法比merge更简洁

# df3 = pd.concat([df1, df2], axis=0) #concat（）函数是一种全连接（UNION ALL)方式，它不需要对齐，而是直接合并，即不需要俩个表的某些列或索引相同，只是把数据整合到一起。
# print(df3)       #因此，该函数没有how和on参数，而是通过axis参数指定连接的轴向，该参数默认为0，按行方向连接，即纵向拼接

# df3 = pd.concat([df1, df2], axis=0, ignore_index=True)  #df3的行索引为原来俩个表各自的索引，若要重置索引，可以在concat函数中设置ignore_index=True,忽略原有索引，生成新的数字序列为索引
# print(df3)

# df3 = pd.concat([df1, df2], axis=1)  #若按列方向连接，即横向拼接，可以设置axis参数为1
# print(df3)

# "append()函数，可以看成concat函数的简化版，效果和pd.concat([df1, df2])类似，代码如下"
# df3 = df1.append(df2)
# print(df3)

# "append()函数，还有个常用的功能，和"列表.append()"一样，可以用来新增元素，代码如下"
df3 = df1.append({'公司': '腾讯', '分数':'90.0'},ignore_index=True)  #这里一定要设置ignore_index=True以忽略原索引，否则会报错
print(df3)