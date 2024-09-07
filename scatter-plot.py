import numpy as np
import matplotlib.pyplot as plt

rng = np.random
x = (rng.rand(50))*10 # มีค่าบวกเท่านั้น
b = rng.randn(50) # มีค่าลบได้
y = 2*x+b

plt.scatter(x,y)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Scatter plot of y=2x+b')
plt.grid()
plt.show()


# y =ax+c