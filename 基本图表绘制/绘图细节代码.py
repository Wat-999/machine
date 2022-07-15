"图名 图例 轴标签 轴边界 轴刻度 轴刻度标签"
import matplotlib
matplotlib.use('TKAgg')  # mac环境下需要加上以上两句，matplotlib才能正常使用。

#解决中文显示问题
from matplotlib.font_manager import *
import matplotlib.pyplot as plt
#定义自定义字体，文件名从查看系统中文字体中来
myfont = FontProperties(fname='/Users/leilei07/Downloads/simheittf-1/simhei.ttf')
#解决负号'-'显示为方块的问题
matplotlib.rcParams['axes.unicode_minus']=False

import numpy as np
import pandas as pd
import matplotlib
# 图名 图例 轴标签 轴边界 轴刻度 轴刻度标签
df = pd.DataFrame(np.random.rand(10,2), columns = ['A','B'])
fig = df.plot(figsize = (10,6))  #创建图表对象，并复制给fig

plt.title('标题',fontproperties=myfont)
plt.xlabel('x轴坐标',fontproperties=myfont)
plt.ylabel('y轴坐标',fontproperties=myfont)
plt.legend(loc = 'upper right')
#   图例放置的位置
#   upper right  右上角
#  upper left    左上角
#  lower left  左下
#  lower right  右下
#  right       右边
#  center left   左中
#  center right  右中
#  lower center   下中心
#  upper center   上中心
#  center         中间
plt.xlim([0,12])  # x轴边界
plt.ylim([0,1.5])  # y轴边界
plt.xticks(range(12))  # 设置x刻度
plt.yticks([0,0.2,0.4,0.6,0.8,1.0,1.2])  # 设置y刻度
fig.set_xticklabels("%.1f" %i for i in range(12))    #x轴刻度标签
fig.set_yticklabels("%.1f" %i for i in [0,0.2,0.4,0.6,0.8,1.0,1.2])  #y轴刻度标签
plt.grid(True,linestyle = "--",color = 'gray' ,linewidth = '0.5',axis='both')
# 显示网格
# linestyle : 线型
# color：颜色
# linewidth ：线宽
# axis = x,y,both,显示x/y/两者的格网
plt.tick_params(bottom = 'on',top = 'off',left = 'on',right = 'off')
#刻度显示
# 刻度分为上下左右四个地方，on为显示刻度，off不显示刻度
plt.axis('off')  #关闭坐标轴
plt.show()

