import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

dataset = pd.read_csv('https://raw.githubusercontent.com/kongruksiamza/MachineLearning/master/Linear%20Regression/Weather.csv')
print(dataset.shape)

# train dataset # test dataset
x = dataset['MinTemp'].values.reshape(-1,1)
y = dataset['MaxTemp'].values.reshape(-1,1)

# 80% 20%
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=0)

# training model
model = LinearRegression()
model.fit(x_train,y_train)

# test model
y_predict = model.predict(x_test)

# compare true data and predict data
df = pd.DataFrame({'True':y_test.flatten(),'Predict':y_predict.flatten()})

# compare loss
print('Mean Absolute Error:',metrics.mean_absolute_error(y_test,y_predict))
print('Mean Squared Error:',metrics.mean_squared_error(y_test,y_predict))
print('Root Mean Squared Error:',np.sqrt(metrics.mean_squared_error(y_test,y_predict)))
print('R2 Score:',metrics.r2_score(y_test,y_predict))


