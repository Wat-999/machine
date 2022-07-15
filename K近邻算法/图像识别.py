#Pollow库是一款功能强大，简单好用的第三方图像处理库
#1图片大小调整及显示
from PIL import Image   #引入pollow库中的Image模块来处理图像
img = Image.open('/Users/macbookair/Desktop/数据分析/书本配套资料及电子书/python/Python大数据分析与机器学习/源代码汇总-2020-12-16/第7章 K近邻算法/源代码汇总_Pycharm/数字4.png')
#open（）函数可以打开JPG、PNG等格式的图片
img = img.resize((64,64))    #resize（）函数可以调整图像大小，这里调整为32*32像素
img.show()   #show（）函数用来显示图片

#2图片灰度处理
#原始图拍片是一个彩色的数字4，对其进行灰度处理，将其转换为黑白的数字4，以便之后将其转换为数字0和1
img = img.convert('L')
img.show()

#3图片二值化处理
#获得黑白的数字4后，就要进行关键的图像二值化处理了
import numpy as np   #引入numpy库，为之后图像转换为二维数组做准备
img_new = img.point(lambda x: 0 if x > 128 else 1)
#point（）函数可以操控每一个像素点，lambda匿名函数，其含义为将色彩数值大于128的像素点赋值为0，反之赋值为1
#图像在进行灰度处理后，每一个像素点由一个取值范围为0～255的数字表示，其中0代表黑色，255代表白色，所以这里以128为阀值进行划分，
#即原来偏白色的区域赋值为0，原来偏黑色的区域赋值为1，这样便完成了将颜色转换成数子0和1的工作
arr = np.array(img_new)  #array（）函数将已经转换成数字0和1的32*32像素的图片转化成32*32的二维数组，并赋给变量arr

#此时可以直接打印出来（print），不过因为其行列较多，可能显示不全所以用循环来一行一行打印
for i in range(arr.shape[0]):  #arr.shape获取的是数组的行数和列数（shape[0]对应行数 shape[1]对应列数）
    print(arr[i])

#4将二维数组 转换成一维数组
#上面获得的32*32二维数组不能用于数据建模，因此还需要用reshape（1，-1）函数将其转换成一行（若写成reshape（-1，1）则转换成一列），即1*1024的一维数组
arr_new = arr.reshape(1, -1)
print(arr_new.shape)   #打印arr_new的行数与列数

#把处理好的一维数组arr_new传入前面训练好的knn模型中
#answer = knn.predict(arr_new)
#print('图片中的数字为：' + str(answer[0]))
#因为获取到的answer是一个一维数组（类似列表），所以通过answer[0]提取其中的元素
#又因为提取的元素是数字，不能直接进行字符串拼接，所以用str（）函数转换后再进行字符串拼接