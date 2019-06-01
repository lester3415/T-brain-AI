import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression as LR
from sklearn import preprocessing

from sklearn.metrics import r2_score

data_train = pd.read_csv("train2.csv")
data_test = pd.read_csv("test.csv")

# print(data_train.corr())

# parking_price = data_train['parking_price'].fillna(43791.94714).values
# print(parking_price)
# land_area = data_train['land_area'].values
# building_area = data_train['building_area'].values
# total_price = data_train['total_price'].values

data_train['parking_price'].fillna(
    data_train['parking_price'].median(), inplace=True)

data_test['parking_price'].fillna(
    data_test['parking_price'].median(), inplace=True)

x_train = data_train[['parking_price', 'land_area', 'building_area',
                      'doc_rate', 'master_rate', 'bachelor_rate', 'jobschool_rate']].values
x_test = data_test[['parking_price', 'land_area', 'building_area',
                    'doc_rate', 'master_rate', 'bachelor_rate', 'jobschool_rate']].values
y = data_train[['total_price']].values
y = np.log(y)

X_train1, X_test1, y_train1, y_test1 = train_test_split(
    x_train, y, test_size=0.3, random_state=0)

regs = LR()
regs.fit(X_train1, y_train1)
print(regs.score(X_test1, y_test1))

# 
real_price = np.expm1(regs.predict(x_test))
print(real_price)

# # produce submit_test.csv
# prediction = pd.DataFrame(real_price, columns=['total_price'])
# result = pd.concat([data_test['building_id'], prediction], axis=1)
# result.to_csv('submit_test.csv', index=False)

plt.scatter(regs.predict(X_test1), y_test1, alpha=0.5)
# plt.plot(X_train1, regs.predict(X_train1), color='lightcoral')
plt.show()
