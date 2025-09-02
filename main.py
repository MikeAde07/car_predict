import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn import metrics


#Data collection and processing

car_data = pd.read_csv('car data.csv')

#encoding "Fuel_Type" Column

car_data.replace({'Fuel_Type': {'Petrol':0, 'Diesel':1, 'CNG':2}}, inplace=True)

#encoding "Seller_Type" Column

car_data.replace({'Seller_Type': {'Dealer':0, 'Individual':1}}, inplace=True)

#encoding "Transmission" Column

car_data.replace({'Transmission': {'Manual':0, 'Automatic':1}}, inplace=True)

#Splitting the data into Features and Target

X = car_data.drop(columns=['Car_Name','Selling_Price'], axis=1)
Y = car_data['Selling_Price']

#Splitting the data into Training data and Test data

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, random_state=2)

#Loading the Linear Regression model

lin_reg = LinearRegression()
lin_reg.fit(X_train, Y_train)

#prediction on Training data
train_pred = lin_reg.predict(X_train)

# R squared error
train_acc = metrics.r2_score(Y_train, train_pred)

#prediction on Testing data
test_pred = lin_reg.predict(X_test)

# R squared error
test_acc = metrics.r2_score(Y_test, test_pred)

print("R squared Error : ", test_acc)

#Visualize the actual prices and predicted prices

plt.scatter(Y_test, test_pred)
plt.xlabel("Actual prices")
plt.ylabel("Predicted prices")
plt.title(" Actual prices vs Predicted prices")
plt.show()

#Lasso Regression

#Loading the Linear Regression model

lasso_reg = Lasso()
lasso_reg.fit(X_train, Y_train)

#prediction on Training data
train_lasso_pred = lasso_reg.predict(X_train)

# R squared error
train_lasso_acc = metrics.r2_score(Y_train, train_lasso_pred)

#prediction on Testing data
test_lasso_pred = lasso_reg.predict(X_test)

# R squared error
test_lasso_acc = metrics.r2_score(Y_test, test_lasso_pred)

print("R squared Error : ", test_lasso_acc)

#Visualize the actual prices and predicted prices

plt.scatter(Y_test, test_lasso_pred)
plt.xlabel("Actual prices")
plt.ylabel("Predicted prices")
plt.title(" Actual prices vs Predicted prices")
plt.show()