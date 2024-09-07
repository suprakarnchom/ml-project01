import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-5,5,10)
a = 2
b = 1
y = a*x + b
# print(x)
# print(y)
plt.plot(x,y,'-r',label='y=2x+1')
plt.xlabel('x')
plt.ylabel('y')
plt.legend(loc='upper left')
plt.title('Graph of y=2x+1')
plt.grid()
plt.show()