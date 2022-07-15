#贝叶斯分类是机器学习中应用极为广泛的分类算法之一。
from sklearn.naive_bayes import GaussianNB    #这里用的是高斯贝叶斯分类器
x = [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]]
y = [0, 0, 0, 1, 1]
model = GaussianNB()
model.fit(x, y)
print(model.predict([[5, 5]]))