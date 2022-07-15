import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
x = [[1], [2], [4], [5]]
y = [2, 4, 6, 8]
poly_reg = PolynomialFeatures(degree=2)
x_ = poly_reg.fit_transform(x)

regr = LinearRegression()
regr.fit(x_, y)

plt.scatter(x, y)
plt.plot(x, regr.predict(x_), color='red')
plt.show()

print(regr.coef_)         #获取系数a、b
print(regr.intercept_)    #获取常数项c
