
# -*- coding: utf-8 -*-
import os
import pandas as pd
#新建列表，存放每个文件数据框（每一个excel读取后存放在数据框）
result = []
path = "/Users/macbookair/Desktop/ox"
for root, dirs, files in os.walk(path, topdown=False):
    for name in files:
        if name.endswith(".xls") or name.endswith(".xlsx"):
            df = pd.read_excel(os.path.join(root, name), sheet_name=None) #excel转换成DataFrame
            result.append(df)

data_list = []
for data in result:
    # print(data.values())
    data_list.extend(data.values())  # 注意这里是extend()函数而不是append()函数

df = pd.concat(data_list)
df.to_excel("/Users/macbookair/Desktop/ox/ax12.xlsx", index=False)
print("合并完成!")