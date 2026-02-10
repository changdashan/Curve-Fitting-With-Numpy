"""
    File:           ModelFitter-Example3.py
    Author:         Dashan Chang
    Created:        1/16/2026    
    Description:    A demontration example on usage of the ModelFitter class. 
                    Partition curves is an important tool in analysis of coal preparation industry.
                    There have been many mathematical models developed for it. 
                    Here I introduced one I've not seen before:
                    Logistics and its variant function.
                    It also demonstrate how to use weights (1/sigma**2) in curve fitting.
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import time
from ModelFitter2 import ModelFitter, curve_fits, minimizes
import warnings

from PartitionCurve import model_Modified_HyperbolicTangent

#--------------------------------------------------------------------------------------------------
# Logistics model
# where x0 is the separation density. k is parameter to scale the axis
def model_Logistics(x, *p):	
    k = p[0]
    x0 = p[1]
    z =  -k * (x-x0)
    if (z.max() < 12 or z.min() > -12):
        f = 100 *  1 /( 1 +  np.exp(z) )
    else:
        print(z)
    return f

#--------------------------------------------------------------------------------------------------
# modified logistics model
def model_Modified_Logistics(x, *p):
    k = p[0]
    x0 = p[1]
    a = p[2]
    b = p[3]
    
    z =  -k * (x-x0) 
    f = 100 * (a + b * x + 1 /( 1 +  np.exp(z) ) )     
    return f

#---------------------------------------------------------------------------------------------------
# example measured data for coal separator
X = np.array([1.25, 1.35, 1.45, 1.55, 1.65, 1.75, 1.90, 2.20])
Y = np.array([0.16, 1.90, 10.84, 34.59, 64.25, 83.68, 95.12, 99.65])
sigma = np.array([0.1, 0.5, 1, 1, 1, 1, 0.5, 0.2 ])  
model = model_Logistics
P = np.array([10, 1.5])
model = model_Modified_Logistics
P = np.array([10, 1.5, 1, -1])

start_time = time.perf_counter()

popt, pcov = curve_fits(model, X, Y, p0=P, sigma=sigma)

print("Optimized parameters:", np.round(popt, decimals=10))
print("variance and covariance of estimated parameters:", pcov)

end_time = time.perf_counter()
elapsed_time = end_time - start_time
print(f"The code block executed in {elapsed_time:.4f} seconds")

#tested and it works with the following scipy.optimize.minimize methods:
#Nelder-Mead, Powell, BFGS, L-BFGS-B, TNC, SLSQP 
for method in ("Nelder-Mead","Powell", "BFGS", "L-BFGS-B", "TNC", "SLSQP", "curve_fit", "curve_fits"):
    print("\r\n")
    print("Method:", method)
    start_time = time.perf_counter()
    minimizeResult = minimizes(model, X, Y, p0=P, sigma=sigma, method=method)
    print("Minimized Result:", minimizeResult)
    print("The estimated parameters:", np.round(minimizeResult.x, decimals=10))    
    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    print(f"The code block executed in {elapsed_time:.4f} seconds with method: {method}")

x0 = np.linspace(1.2, 2.4, 120)
y0 = model(x0, *popt)

x = []
y = []

for i in range(len(x0)):
    if y0[i] >= 0 and y0[i] <= 100 :
        x.append(x0[i])
        y.append(y0[i])        

fig, ax = plt.subplots()

plt.scatter(X,Y,label='Data')
plt.grid(color='gray', linestyle='--', linewidth=0.5)
plt.plot(x, y, 'r-', label='Fit')
plt.legend()
plt.xlabel("Coal Density")
title = model.__name__
plt.ylabel("Partition (%)")
plt.title("Model: " + title)
xtick_positions = [1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4]
plt.xticks(xtick_positions)  
ytick_positions = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
plt.yticks(ytick_positions)    
    
manager = plt.get_current_fig_manager()
manager.window.title("Coal Partition Curve")

plt.show()

print("\r\n")
print("Measured Y[i]:", Y)
y = model(X, *popt)
print ("Predicted y[i]:", y)
loss = (Y - y) / sigma
print("loss[i]:", loss)
ss = loss ** 2
print("square of loss[i]]:", ss)
print("sum of squares of loss]:", ss.sum())
s = np.sqrt(ss.sum()/len(Y))
print("Square Root of the sum of squares:", s)
