# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
#### 1.Import the standard Libraries. 
#### 2.Set variables for assigning dataset values. 
#### 3.Import linear regression from sklearn. 
#### 4.Assign the points for representing in the graph. 
#### 5.Predict the regression for marks by using the representation of the graph. 
#### 6.Compare the graphs and hence we obtained the linear regression for the given datas.
## Program:
```py
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: MOHAMMAD FAIZAL SK
RegisterNumber: 212223240092
*/
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
df=pd.read_csv('student_scores.csv')
print(df)
df.head(0)
df.tail(0)
print(df.head())
print(df.tail())
x = df.iloc[:,:-1].values
print(x)
y = df.iloc[:,1].values
print(y)
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)
y_pred = regressor.predict(x_test)
print(y_pred)
print(y_test)
#Graph plot for training data
plt.scatter(x_train,y_train,color='black')
plt.plot(x_train,regressor.predict(x_train),color='blue')
plt.title("Hours vs Scores(Training set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
#Graph plot for test data
plt.scatter(x_test,y_test,color='black')
plt.plot(x_train,regressor.predict(x_train),color='red')
plt.title("Hours vs Scores(Testing set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
mse=mean_absolute_error(y_test,y_pred)
print('MSE = ',mse)
mae=mean_absolute_error(y_test,y_pred)
print('MAE = ',mae)
rmse=np.sqrt(mse)
print("RMSE= ",rmse)
```
## Output:
### Dataset
![Screenshot 2024-08-30 123756](https://github.com/user-attachments/assets/2835b5e4-77e3-4eba-92ef-fd30b7da4b8a)

### Head Values
![image](https://github.com/user-attachments/assets/23a7d686-86aa-4444-8443-423ff6d8fb83)

### Tail Values
![image](https://github.com/user-attachments/assets/8247ffaf-1261-4368-92e2-10a805dee7d2)

### X and Y values
![image](https://github.com/user-attachments/assets/d3266830-d2b6-4423-8420-bf4d111538e9)


### Predication values of X and Y
![image](https://github.com/user-attachments/assets/e98017bf-f708-486f-80f7-80ef7e300166)

### MSE,MAE and RMSE
![image](https://github.com/user-attachments/assets/3445c487-696f-417c-9d44-d32dcd727913)

### Training Set
![image](https://github.com/user-attachments/assets/bc5883a1-7136-43db-a1f0-53fd3fbc7e64)

### Testing Set
![image](https://github.com/user-attachments/assets/e73947f6-ed6e-4a51-8e5c-85eae472979d)

## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
