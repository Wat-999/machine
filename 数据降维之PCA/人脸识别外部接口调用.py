# # 12.3 补充知识点：人脸识别外部接口调用   #百度人脸识别
from aip import AipFace
import base64
import chardet
APP_ID = '26001964'
API_KEY = 'c2c9eowWG5qCdhGEvsHL1QqG'
SECRET_KEY = 'WSp6evpK5XKQtHkLt8MTohUZhU8CVauI'
#把上述参数传入接口
aipFace = AipFace(APP_ID, API_KEY, SECRET_KEY)
filePath = r'/Users/macbookair/Desktop/数据分析/书本配套资料及电子书/python/Python大数据分析与机器学习/源代码汇总-2020-12-16/第12章 数据降维之PCA主成分分析/源代码汇总_PyCharm格式/王宇韬.jpg'
def get_file_content(filePath):
    with open(filePath, 'rb') as fp:
        content = base64.b64encode(fp.read())
        return content.decode('utf-8')
imageType = "BASE64"

options = {}
options["face_field"] = "age, gender, beauty"

result = aipFace.detect(get_file_content(filePath), imageType, options)
print(result)
age = result['result']['face_list'][0]['age']
print('年龄预测为：' + str(age))
gender = result['result']['face_list'][0]['gender']['type']
print('性别预测为：' + gender)
beauty = result['result']['face_list'][0]['beauty']
print('颜值评分为：' + str(beauty))


