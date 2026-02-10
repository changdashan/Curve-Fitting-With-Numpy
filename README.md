	Description: 
  
		ModelFitter is a Python class designed to fit a model function of the form f(x, p) to a set of measured data points (X, Y).
		The class is implemented purely via numpy's matrix manipulations
		ModelFitter employs the Levenberg-Marquardt algorithm to iteratively update the model parameters.
	    A damping factor (lambda) is applied to the diagonal elements of the normal equation matrix to control the gradient descent step size and improve convergence stability.
		To further refine each iteration step, the class uses a Golden Section Search line-search method to determine an optimal step length along the descent direction.
		The fitting process proceeds iteratively as follows:
			1. Build the Jacobian matrix by numerically estimating partial derivatives.
			2. Construct the normal equations A as (J^T * W * J) with Levenberg-Marquardt damping.
			3. Solve the normal equations with numpy.linalg.solve.
			4. Perform a Golden Section Search to determine the optimal step size.
			5. Update the parameters and repeat until convergence is achieved.
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
