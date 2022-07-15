"添加文字说明"
import matplotlib.pyplot as plt
x = [1, 2, 3]
y = [2, 4, 6]
plt.plot(x, y)    #绘制折线图
plt.title('TITLE')  #添加图表标题
plt.xlabel('x')     #添加x轴标签
plt.ylabel('y')     #添加y轴标签
plt.show()

"添加图例"
import numpy as np
import matplotlib.pyplot as plt
#第一条线：设置label（标签）为"y = x + 1"
x1 = np.array([1, 2, 3])   #np.array([列表])为创建数组
y1 = x1 + 1
plt.plot(x1, y1, label='y = x + 1')  #label设置标签
#第二条线：设置label（标签）为"y = x*2"
y2 = x1*2
plt.plot(x1, y2, color='red', linestyle='--', label='y = x*2')
#设置图例位置为左上角，若要修改图例位置修改loc参数的值即可，'upper right'代表右上角，'lower right'代表右下角
plt.legend(loc='upper left')
plt.show()

"设置双坐标轴"
import numpy as np
import matplotlib.pyplot as plt
#第一条线：设置label（标签）为"y = x + 1"
x1 = np.array([10, 20, 30])   #np.array([列表])为创建数组
y1 = x1
plt.plot(x1, y1, label='y = x ')  #label设置标签
plt.legend(loc='upper left')    #设置该图表图例在左上角
plt.twinx()     #设置双坐标轴
#第二条线：设置label（标签）为"y = x^2"
y2 = x1*x1
plt.plot(x1, y2, color='red', linestyle='--', label='y = x^2')
#设置图例位置为左上角，若要修改图例位置修改loc参数的值即可，'upper right'代表右上角，'lower right'代表右下角
plt.legend(loc='upper right')   #设置该图表图例在右上角
plt.show()

"设置图表大小、设置x轴刻度的角度"
import numpy as np
import matplotlib.pyplot as plt
#第一条线：设置label（标签）为"y = x + 1"
x1 = np.array([10, 20, 30])   #np.array([列表])为创建数组
y1 = x1
plt.plot(x1, y1, label='y = x ')  #label设置标签
plt.legend(loc='upper left')    #设置该图表图例在左上角
plt.twinx()     #设置双坐标轴
#第二条线：设置label（标签）为"y = x^2"
y2 = x1*x1
plt.plot(x1, y2, color='red', linestyle='--', label='y = x^2')
#设置图例位置为左上角，若要修改图例位置修改loc参数的值即可，'upper right'代表右上角，'lower right'代表右下角
plt.legend(loc='upper right')   #设置该图表图例在右上角
plt.rcParams['figure.figsize'] = (8, 6)  #设置图表大小，第一个元素代表长即800像素，第二个元素代表宽即600像素
plt.xticks(rotation=45)  #若x轴因为刻度内容较多，导致刻度太密，可以通过设置刻度的角度来进行调节，其中45即为45度
plt.show()

"绘制多图"
import matplotlib.pyplot as plt
#subplot(221)函数，它的参数通常表示为3位数，该整数的各位数字分别代表子图的行数、列数及当前子图的序号。
#subplot(221)表示绘制2行2列的子图（共4张子图），ax1表示为第一张图
ax1 = plt.subplot(221)
plt.plot([1, 2, 3], [2, 4, 6])
ax2 = plt.subplot(222)
plt.bar([1, 2, 3], [2, 4, 6])
ax3 = plt.subplot(223)
plt.scatter([1, 3, 5], [2, 4, 6])
ax4 = plt.subplot(224)
plt.hist([2, 2, 2, 3, 4])
plt.show()

"绘制多图简洁写法"
"subplots()函数主要有俩个参数：nrows表示行数，ncols表示列数。可以简写成plt.subplots(2, 2）就表示绘制2行2列共4张子图" \
"它会返回两个内容：fig（画布）和axes（子图集合，以数组形式存储各张子图）" \
"第2行代码使用flatten（）函数将子图展开，从而获得各张子图"
fig, axes = plt.subplots(2, 2, figsize=(10, 8))   #figsize参数设置图表尺寸为1000X800像素
ax1, ax2, ax3, ax4 = axes.flatten()
ax1.plt.plot([1, 2, 3], [2, 4, 6])  #绘制第一张图
ax2.plt.bar([1, 2, 3], [2, 4, 6])   #绘制第二张图
ax3.plt.scatter([1, 3, 5], [2, 4, 6])  #绘制第三张图
ax4.plt.hist([2, 2, 2, 3, 4])         #绘制第四张图
plt.show()