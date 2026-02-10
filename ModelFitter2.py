"""
	File: ModelFitter2.py
	Author:		Dashan Chang
	Date:		02/10/2025

	Description: 
		ModelFitter is a Python class designed to fit a model function of the form f(x, p) to a set of measured data points (X, Y).
		The class is implemented purely via numpy's matrix manipulations
		ModelFitter employs the Levenberg-Marquardt algorithm to iteratively update the model parameters.
	    A damping factor (lambda) is applied to the diagonal elements of the normal equation matrix to control the gradient descent step size and improve convergence stability.
		To further refine each iteration step, the class uses a Golden Section Search line-search method to determine an optimal step length along the descent direction.
		The fitting process proceeds iteratively as follows:
			1. Build the Jacobian matrix by numerically estimating partial derivatives.
			2. Construct the normal equations A as (J^T * W * J) with Levenberg-Marquardt damping.
			3. Construct the right hand side of the equation as (J^T * W * dy)
			4. Solve the normal equations with numpy.linalg.solve.
			5. Perform a Golden Section Search to determine the optimal step size.
			6. Update the parameters and repeat until convergence is achieved.
		The iteration stops when the relative parameter change satisfies: ((p[k+1] - p[k]) /  p[k]) < 1e-6 
		For ease of use, two convenience functions called curve_fits and minimize are provided
		and can be imported directly into client code to perform model fitting without interacting with the class internals.
			
	Usage:	
	
		from ModelFitter import ModelFitter, curve_fits, minimizes
			
		def model_antelope_population(x, *p):
			a = p[0]
			k = p[1]
			f = a * np.exp(k * x)
			return f

		T = np.array([1,2,4,5,8])
		Y = np.array([3,4,6,11,20])
		P = np.array([2, 1])
		model = model_antelope_population
		popt, pcov = curve_fits(model, T, Y, p0=P)
		print(popt)
		print(pcov)
				
"""

import numpy as np
from scipy.optimize import minimize, curve_fit, OptimizeResult

class ModelFitter:
	
	def __init__(self, X, Y, ModelFunc, P, sigma=None, DerivativeFunc=None, MaxIterations=50, ftol=1e-6, xtol=1e-6):
		self.X = X
		self.Y = Y
		self.Sigma = np.array([1] * len(X)) if sigma is None else sigma								# assume sigma is 1 if not provided. inverse of weight, 1/sigma
		self.Model = ModelFunc
		self.P = P
		self.Derivative = self.DerivativeEstimate if DerivativeFunc is None else DerivativeFunc		# user does not need to provide it. We will use numberic method to get the partial derivative
		self.PP = None																				# Previous Parameter or parameters estimated at the current iteration
		self.DP = None																				# Delta Parameters
		self.epsilon = 0.000001 if xtol != 0 else xtol
		self.MaxIterations = 50 if MaxIterations == 0 else MaxIterations
		self.Iteration = 0
		self.x = None
		self.y = None
		self.xError = None
		self.yError = None
		self.xStandardDeviation = None
		self.yStandardDeviation = None
		self.ModelName = ModelFunc.__name__
		self.ModelFormula = ""
		self.ParameterNames = "" 
		self.lamda = 1
		self.pcov = None
		self.popt = None
		self.ObjectiveFunction = None
	

	#-----------------------------------------------------------------------------------------
	def DerivativeEstimate(self, x, P):
		ph = P.copy()
		m = len(P)
		df = [0] * m
		for i in range(m):
			p = ph[i]
			h = 0.0001
			ph[i] = p + h
			f1 = self.Model(x, *ph)
			ph[i] = p - h
			f2 = self.Model(x, *ph)
			df[i] = (f1 - f2)/(2.0 * h)            # Three-point central difference formula
		return df

	#---------------------------------------------------------------------------------------- 
	# get the square root of the sum of the squares of the residual/difference/deviation	
	# By adding the parameter p, this method can be used as an objective function for all minimize methods in scipy.optimize
	def SqrtOfSumOfSquares(self, p=None):
		ss = self.SumOfSquares(p)		
		sss = np.sqrt(ss / len(self.Y))
		return sss

	#----------------------------------------------------------------------------------------
	# the objective function to be minimized 
	# calculate the square root of the sum of the squares of the residual/difference/deviation	
	# By adding the parameter p, this method can be used as an objective function for all minimize methods in scipy.optimize
	def SumOfSquares(self, p=None):
		if p is None:
			p = self.P
		s = self.Y - self.Model(self.X, *p)
		ss = np.power(s, 2).sum()
		return ss

	#-----------------------------------------------------------------------------------------
	# constructing Jacobian matrix with the partial derivatives of the sum of the residual squares 
	# with respect to each model parameter at each X point, which is an n * m matrix  
	# The returned Jacobian will be used for pure matrix operations
	def GetJacobian(self):
		m = len(self.P)											# number of parameters of the model
		n = len(self.X)											# number of measured x values

		J = [0] * n												# J - Jacobian matrix		
		for i in range(n):
			J[i] = [0] * m
			x = self.X[i]
			y = self.Y[i]
			p = self.P
			df = [0] * m
			df = self.Derivative(x, p)							# df - the partial derivative of the model with respect to each parameter.
			for j in range(m):
				J[i][j] = df[j]
		return J
	
	#-----------------------------------------------------------------------------------------
	# Using numpy's matrix manipulations to solve for the gradient descening step size, i.e., delta P (or beta) at one interation 
	# Please refer to Non-linear least squares on Wikimedia.
	def SolveForDeltaP(self):						
		J = np.array(self.GetJacobian())							
		Jt = J.T		
		Sgma = np.diag(self.Sigma ** 2)								# sigma has to be squared as w = 1 / np.power(sigma, 2)
		W = np.linalg.inv(Sgma)										# W is the weight for each measured data
		dy = self.Y - self.Model(self.X, *self.P)					# residual of measured and predicted data
		B = Jt @ W @ dy												# B vector with sigma
		A = Jt @ W @ J												# A m * m normal equation matrix
		LambdaI = np.identity(len(A)) * self.lamda					# create an identity matrix and multiply the lambda (the Levenburgh-Marquardt factor)
		A += LambdaI												# apply the LM factor to the diagonal elements of A
		dp = np.linalg.solve(A, B)									# solve the normal equations for the delta P (delta Beta), gradient descending step size
		return dp

	#----------------------------------------------------------------------------------------
	# Golden-Section Search, one of the line search algorithms to determine 
	# the next set of parameters so that the objective function is descending the fastest.
	def GoldenSectionSearch(self):
		b1 = 0
		b2 = 1
		c = 0.5 * (np.sqrt(5) - 1)         #0.618
		a2 = b1 + (b2 - b1) * c
		
		m = len(self.P)

		self.PP = self.P.copy()
		p0 = self.PP
		dp = self.DP
		p = [0] * m

		for i in range(m):
			p[i] = p0[i] - a2 * dp[i]

		self.P = p
		f2 = self.SumOfSquares()

		a1 = b1 + b2 - a2
		for i in range(m):
			p[i] = p0[i] + a1 * dp[i]

		self.P = p
		f1 = self.SumOfSquares()

		while (np.abs(f2 - f1) / (f2 + f1)) > 0.05:
			if f1 < f2:
				b2 = a2
				a2 = a1
				f2 = f1
				a1 = b1 + b2 - a2
				for i in range(m):
					p[i] = p0[i] + a1 * dp[i]

				self.P = p
				f1 = self.SumOfSquares()
			else:
				b1 = a1
				a1 = a2
				f1 = f2
				a2 = b1 + b2 - a1
				for i in range(m):
					p[i] = p0[i] + a2 * dp[i]

				self.P = p
				f2 = self.SumOfSquares()

		goldenStep = 0.5 * (b1 + b2) 
		for i in range(m):
			p[i] = p0[i] + goldenStep * dp[i]						#p[k+1] = p[k] + f * dp   (aka, delta Beta, which is solved by GaussianEleminationMethod), a method called Shift-butting.

		return p
	
	#----------------------------------------------------------------------------------------
	def Minimize(self, method="Powell"):
		p0 = self.P
		ObjectiveFunc = self.SumOfSquares
		result = minimize(ObjectiveFunc, p0, method=method)
		return result
		
	#-----------------------------------------------------------------------------------------
	# Calculate the variances and covariances of the estimated optimal parameters
	def CalcParamVariance(self):
		J = np.array(self.GetJacobian())							
		Jt = J.T		
		Sgma = np.diag(self.Sigma ** 2)								# sigma has to be squared as w = 1 / np.power(sigma, 2)
		W = np.linalg.inv(Sgma)										# W is the weight for each measured data
		A = Jt @ W @ J
		inv_A = np.linalg.inv(A)	
		Q = self.SumOfSquares()
		N = len(self.X)
		M = len(self.P)
		rv = Q/(N-M)
		pcov = inv_A * rv
		self.pcov = pcov
		return pcov

	#-----------------------------------------------------------------------------------------
	# The interation is using matrix manipulations of numpy to solve the delta P (sometime also called beta).
	def Iterate(self):	
		epsilon = self.epsilon
		k = 0
		lamda = 1.0
		Converged = False
		MaxIterations = self.MaxIterations

		np.set_printoptions(suppress=True, precision=8)

		while not Converged and k <= MaxIterations:
			k += 1		
			self.Iteration = k
			dp = self.SolveForDeltaP()
			self.DP = dp
			
			p = self.P
			epsilon = self.epsilon
			
			Converged = True
			for i in range(len(p)):
				epsiln = np.abs(dp[i] / p[i])				
				if epsiln > epsilon:
					Converged = False

			if not Converged:
				p = self.GoldenSectionSearch()
				self.P = p
				self.lamda *= 0.8

		leastResidual = self.SumOfSquares()
		self.popt = self.P
		
		print("The Least Residual Error: ", leastResidual)
		for i in range(len(p)):
			print(f"Parameter: {i} value: {p[i]}")

		self.CalcParamVariance()
		

#---------------------------------------------------------------------------------------------------
# simulate the curve_fit method of scipy.optimize
def curve_fits(Model, X, Y, p0=None, sigma=None, Derivative=None, MaxIterations=50, ftol=1e-6, xtol=1e-6, full_output=False):
	modelFitter = ModelFitter(X, Y, Model, p0, sigma, Derivative, MaxIterations, ftol, xtol)
	modelFitter.Iterate()
	popt = modelFitter.popt
	pcov = modelFitter.pcov
	if full_output:
		return popt, pcov, modelFitter
	else:
		return popt, pcov
	
#---------------------------------------------------------------------------------------------------
# simulate the minimize method of scipy.optimize
def minimizes(Model, X, Y, p0=None, sigma=None, Derivative=None, MaxIterations=50, ftol=1e-6, xtol=1e-6, full_output=False, method="Powell"):
	if method == "curve_fit":
		popt, pcov = curve_fit(Model, X, Y, p0=p0, sigma=sigma, full_output=False)
		minimizeResult = OptimizeResult(x=popt, hess_inv=pcov)
	elif method == "curve_fits":
		popt, pcov, modelFitter = curve_fits(Model, X, Y, p0=p0, sigma=sigma, full_output=True)
		minimizeResult = OptimizeResult(x=popt,hess_inv = pcov)
	else:
		modelFitter = ModelFitter(X, Y, Model, p0, sigma, Derivative, MaxIterations, ftol, xtol)		
		minimizeResult = modelFitter.Minimize(method)
	return minimizeResult
  
