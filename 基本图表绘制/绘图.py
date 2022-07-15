"绘制折线图"
import matplotlib.pyplot as plt
x = [1, 2, 3]
y = [2, 4, 6]
plt.plot(x, y)   #绘制折线图
plt.show();      #展示图形

import numpy as np
import matplotlib.pyplot as plt
x1 = np.array([1, 2, 3])
y1 = x1 + 1      #第一条线
plt.plot(x1, y1)
y2 = x1*2   #第二条线
#color为颜色，linewidth为线宽，单位为像素，linestyle为线型，默认为实线，"--"表示为虚线
plt.plot(x1, y2, color='red', linewidth=3, linestyle='--')
plt.show()

"绘制柱形图"
import matplotlib.pyplot as plt
x = [1, 2, 3, 4, 5]
y = [5, 4, 3, 2, 1]
plt.bar(x, y)   #绘制柱形图
plt.show()

"绘制散点图"
import matplotlib.pyplot as plt
import numpy as np
x = np.random.rand(10)   #np.random.rand（10）生成10个随机数
y = np.random.rand(10)
plt.scatter(x, y, color='red')    #使用plt.scatter（）函数绘制散点图
plt.show()

"绘制直方图"
import matplotlib.pyplot as plt
import numpy as np
data = np.random.rand(10000)  #随机生成10000个数据
#绘制频数直方图，bins为颗粒度，即直方图的柱形数量，edgecolor为柱形的边框颜色
plt.hist(data, bins=40, edgecolor='black')  #plt.hist为绘制直方图
plt.show()

"运用pandas绘制组合图，本质上还是通过pandas库调用matplotlib库，注意只适用于pandas库创建的dataframe，不适用于numpy库创建的数组"

import matplotlib.pyplot as plt   #如果绘图过程中出现中文乱码，可以在最前面加上如下代码
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']   #用来正常显示中文标签
import pandas as pd
df = pd.DataFrame([[8000, 6000], [7000, 5000], [6500, 4000]],
    columns=['人均收入', '人均支出'], index=['北京', '上海', '广州'])
#df['人均收入'].plot(kind='line', color='red')  #设置kind为line表示绘制折线图
#df['人均收入'].plot(kind='bar')   #设置kind为bar表示绘制柱形图
#plt.show()
# df['人均收入'].plot(kind='pie')   #设置kind参数为pie，则可以绘制饼图
df['人均收入'].plot(kind='box')     #设置kind参数为box，则可以绘制箱体图
plt.show()



