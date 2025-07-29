import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

X = np.array([[50],[70],[100]])
y = np.array([1500000,200000,300000])

model = LinearRegression()
model.fit(X,y)

new_data = np.array([[80]])
estimated = model.predict(new_data)
print("estimated price:" estimated[0])

print("slope", model.coef_[0])
print("Y-axis intersection",model.intercept_)

plt.scatter(X, y, color = 'blue' , label ='real data')
plt.plot(X, model.predict(X) , color = 'red' , label = 'linear model')
plt.xlabel("square meter")
plt.ylabel("price")
plt.legend()
plt.show()

