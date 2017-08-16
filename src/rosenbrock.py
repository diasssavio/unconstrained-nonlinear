import numpy as np
from scipy.optimize import minimize

def rosen(x):
    """Generalised Rosenbrock function"""
    return sum(100.0 * (x[1:] - x[:-1] ** 2.0) ** 2.0 + (1 - x[:-1]) ** 2.0)

def penal(x):
    """Extended Penalty function"""
    return sum((x[:-1] - 1) ** 2) + (sum(x[1:] ** 2 - 0.25)) ** 2

# x0 = np.array([-1.2, 1]) # An initial solution in 2 dimensions
# res = minimize(rosen, x0, method='nelder-mead', options={'xtol': 1e-8, 'disp': True})

print rosen(np.array([1,1]))

x0 = np.array([1,2,3,4])
res = minimize(penal, x0, method='nelder-mead', options={'xtol': 1e-8, 'disp': True})

if res.status == 0:
    print 'Minimum objective value: ', res.fun
    print 'Solution: ', res.x
