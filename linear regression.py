#Simple Linear Regression

#Nhập packages
import matplotlib.pyplot as plt
import pandas as pd
import pylab as pl
import numpy as np

#Đọc data từ tệp FuelConsumptionCo2.csv
df = pd.read_csv("FuelConsumptionCo2.csv")
df.head()
df.describe()

#Lọc dữ liệu khai thác tiếp
cdf = df[['ENGINESIZE','CO2EMISSIONS']]
cdf.head(9)

#Vẽ tương quan
plt.scatter(cdf.ENGINESIZE, cdf.CO2EMISSIONS,  color='blue')
plt.xlabel("Engine size")
plt.ylabel("Emission")
plt.show()

#Vẽ histogram
viz = cdf[['ENGINESIZE','CO2EMISSIONS']]
viz.hist()
plt.show()

#Tạo tập dữ liệu huấn luyện và thử
msk = np.random.rand(len(cdf)) < 0.8
train = cdf[msk]
test = cdf[~msk]

#Vẽ tập huấn luyện
plt.scatter(train.ENGINESIZE, train.CO2EMISSIONS,  color='blue')
plt.xlabel("Engine size")
plt.ylabel("Emission")
plt.show()

#Nhập mô hình hồi quy tuyến tính dựa trên bình phương tối thiểu
from sklearn import linear_model
regr = linear_model.LinearRegression()
train_x = np.asanyarray(train[['ENGINESIZE']])
train_y = np.asanyarray(train[['CO2EMISSIONS']])
regr.fit (train_x, train_y)

#Hệ số và vẽ đường hồi quy, xuat file .eps
print ('Coefficients: ', regr.coef_)
print ('Intercept: ',regr.intercept_)
plt.scatter(train.ENGINESIZE, train.CO2EMISSIONS,  color='blue')
plt.plot(train_x, regr.coef_[0][0]*train_x + regr.intercept_[0], '-r')
plt.xlabel("Engine size")
plt.ylabel("Emission")
plt.savefig('fig.eps', format='eps')

#tinh sai so
from sklearn.metrics import r2_score
test_x = np.asanyarray(test[['ENGINESIZE']])
test_y = np.asanyarray(test[['CO2EMISSIONS']])
test_y_ = regr.predict(test_x)

print("Mean absolute error: %.2f" % np.mean(np.absolute(test_y_ - test_y)))
print("Residual sum of squares (MSE): %.2f" % np.mean((test_y_ - test_y) ** 2))
print("R2-score: %.2f" % r2_score(test_y , test_y_) )

