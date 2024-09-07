import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

rng = np.random
# การจำลองข้อมูล
x = (rng.rand(50))*10 # มีค่าบวกเท่านั้น
b = rng.randn(50) # มีค่าลบได้
y = 2*x+b
# print(x)
x_new = x.reshape(-1,1)
# print(x_new)
# linear regression model
model = LinearRegression()
# train model
model.fit(x_new,y)

# test model
xfit = np.linspace(-1,11)
xfit_new = xfit.reshape(-1,1)

yfit = model.predict(xfit_new)
# anlysis model and result
plt.scatter(x,y)
plt.plot(xfit,yfit)
plt.show()




# y =ax+c