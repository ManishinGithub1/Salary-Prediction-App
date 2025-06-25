import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle

data=pd.read_csv(r'C:\Users\ymani\Full Stack Data Science\Resume Projects\Machine Learning\Salary_Prediction_App\Salary_Data.csv')


x=data.iloc[:,:-1]
y=data.iloc[:,-1]

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.8,random_state=0)

from sklearn.linear_model import LinearRegression
regressor=LinearRegression()

regressor.fit(x_train,y_train)

y_pred=regressor.predict(x_test)

comparision=pd.DataFrame({'Actual':y_test,'Predicted':y_pred})
print(comparision)

plt.scatter(x_test,y_test,color='red')
plt.plot(x_train,regressor.predict(x_train),color='blue')
plt.title("Salary vs Experience (test set)")
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

plt.scatter(x_train,y_train,color='red')
plt.plot(x_train,regressor.predict(x_train),color='blue')
plt.title("Salary vs Experience (train set)")
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

y_12=regressor.predict([[12]])
y_20=regressor.predict([[20]])
print(f"Salary for 12 years of experience: {y_12[0]:,.2f}")
print(f"Salary for 20 years of experience: {y_20[0]:,.2f}")

from sklearn.metrics import mean_squared_error

bias=regressor.score(x_train,y_train) #Training scorer^2
variance=regressor.score(x_test,y_test)#Testing score r^2
train_mse=mean_squared_error(y_train,regressor.predict(x_train))#Training mse
test_mse=mean_squared_error(y_test,regressor.predict(x_test))#Testing mse

print(f"Training Score: {bias:.2f}")
print(f"Testing Score (R^2): {variance:.2f}")
print(f"Training MSE: {train_mse:.2f}")
print(f"test MSE: {test_mse:.2f}")

import pickle
filename='linear_model.pkl'
with open(filename,'wb') as file:
    pickle.dump(regressor,file)
print("Model has been saved as linear_model.pkl")

import os
print(os.getcwd())


