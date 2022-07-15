# -*- coding: utf-8 -*-
import glob2 as glob   #目前只有glob2的包，注意给别名glob
import pandas as pd

path = "/Users/macbookair/Desktop/ox"
data = []
for excel_ox in glob.glob(f'{path}/**/[!~]*.xls*'):     #正则表达式  注意Excel_ox（跟的文件夹名）
    # for excel_file in glob.glob(f'{path}/[!~]*.xlsx'):
    excel = pd.ExcelFile(excel_ox)
    for sheet_name in excel.sheet_names:
        df = excel.parse(sheet_name)
        data.append(df)
# print(data)

df = pd.concat(data, ignore_index=True)
df.to_excel("/Users/macbookair/Desktop/ox/多张表合并.xlsx", index=False)
print("合并完成!")


"~~~~~~~~~~~~~简洁方法2～～～～～～～～～～～～～～"
# -*- coding: utf-8 -*-
import glob2 as glob
import pandas as pd

path = "/Users/macbookair/Desktop/ox"
data = []
# for excel_file in glob.glob(f'{path}/**/[!~]*.xlsx'):#其中**代表的是文件夹下的子文件递归。另外就是.xls*了，这个是正则写法，
for excel_ox in glob.glob(f'{path}/[!~]*.xlsx'):# 表示的是既可以处理xls格式，也可以处理xlsx格式的Excel文件，真是妙哉！
    dfs = pd.read_excel(excel_ox, sheet_name=None).values()  #sheet_name=None这个参数带上，代表获取Excel文件中的所有sheet表，
    # 其返回的是一个字典，所有在后面遍历的时候，是以字典的形式进行取值的，效率比前面的方法都要高一些。
    data.extend(dfs)
# print(data)


df = pd.concat(data, ignore_index=True)
df.to_excel("/Users/macbookair/Desktop/ox/--简洁--所有表合并.xlsx", index=False)
print("合并完成!")