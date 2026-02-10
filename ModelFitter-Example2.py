"""
    File:           ModelFitter-Example2.py
    Author:         Dashan Chang
    Created:        1/16/2026    
    Description:    A demontration example on usage of the ModelFitter class.               
"""

from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
from ModelFitter2 import ModelFitter, curve_fits
import time

def model_sin(x, *p):
    a = p[0]
    b = p[1]
    c = p[2]
    f = a * np.sin(b * x) + c * x
    return f


### main func ###

# nonlinear function
X = np.linspace(0.1, 10, 1000)
F = 5 * np.sin(2 * X) + X

noise = np.random.normal(0, 2, len(X))
F = F + noise                               # added some noise to the data calculated by the function

print(len(X))
fig, ax = plt.subplots()
ax.plot(X, F)
plt.show()

X_train, X_test, Y_train, Y_test = train_test_split(X, F, test_size=0.1, random_state=42, shuffle=True)
fig,ax=plt.subplots()
ax.scatter(X_train,Y_train)
#plt.show()

X = X_train
Y = Y_train
P = np.array([10, 2.2, 10])         # initial guess of parameters
model = model_sin

start_time = time.perf_counter()
sigma = []
popt, pcov = curve_fits(model, X, Y, P)

end_time = time.perf_counter()
elapsed_time = end_time - start_time
print(f"The code block executed in {elapsed_time:.4f} seconds")

x = np.linspace(0.1, 10, 1000)
y = model(x, *popt)                 # model predicted values
ax.scatter(x, y)
plt.show()
