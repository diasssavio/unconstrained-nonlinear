import numpy as np
from testF import power, sugar

cont = {'fun':0, 'grad':0} # counter of function and gradient evaluations

def cauchy_LS(func, x):
    ''' Cauchy Line Search method for quadratic functions '''
    # Calculating Hessian matrix
    aux1 = np.eye(len(x))
    aux, _ = func(np.ones(len(x)), mode = 1, counter = cont)
    aux = aux.reshape(len(x), 1)
    hessian = np.array(aux1 * aux)

    grad, _ = func(x, mode = 1, counter = cont)

    return - np.dot(grad, - grad) / np.dot(grad, np.dot(hessian, grad))

def cauchy_LS2(func, grad, direction):
    ''' Cauchy Line Search method for quadratic functions given A '''
    return - np.dot(grad, direction) / np.dot(direction, func(direction, 1, cont)[0] - func(np.zeros(len(direction)), 1, cont)[0])
    # return - np.dot(grad, direction) / np.dot(direction, np.dot(A, direction))

def armijo_LS(func, x, d, t0, theta, sigma):
    ''' Armijo Line Search method for descent methods '''
    t, k = t0, 1
    func_x, _ = func(x, mode = 0, counter = cont)
    grad_x, _ = func(x, mode = 1, counter = cont)
    while func(x + t * d, mode = 0, counter = cont)[0] > func_x + sigma * np.dot(grad_x, t * d):
        # print 't_', k, '=', t
        k += 1
        t *= theta
    return t

def extrapolation(R_k, rho = 2):
    return rho * R_k

def interpolation(L_k, R_k, theta = 0.5):
    return theta * L_k + (1 - theta) * R_k

def goldstein_LS(func, x, d, t0, sigma_A, sigma_C):
    ''' Goldstein Line Search stepsize method for descent optimisation methods '''
    k, L_k, R_k, t_k = 0, 0, 0, t0
    func_x, _ = func(x, mode = 0, counter = cont)
    grad_x, _ = func(x, mode = 1, counter = cont)

    while True:
        func_td, _ = func(x + t_k * d, mode = 0, counter = cont)
        if func_td <= func_x + sigma_A * np.dot(grad_x, t_k * d):
            if ((func_td - func_x) / t_k) >= sigma_C * np.dot(grad_x, d):
            # if np.dot(func(x + t_k * d, mode = 1, counter = cont)[0], d) >= sigma_C * np.dot(grad_x, d):
                return t_k
            else:
                L_k = t_k
        else:
            R_k = t_k

        if R_k == 0:
            t_k = extrapolation(t_k)
        else:
            t_k = interpolation(L_k, R_k)
        # k += 1

def wolfe_LS(func, x, d, t0, sigma_A, sigma_C):
    ''' Wolfe Line Search stepsize method for descent optimisation methods '''
    k, L_k, R_k, t_k = 0, 0, 0, t0
    func_x, _ = func(x, mode = 0, counter = cont)
    grad_x, _ = func(x, mode = 1, counter = cont)

    while True:
        if func(x + t_k * d, mode = 0, counter = cont)[0] <= func_x + sigma_A * np.dot(grad_x, t_k * d):
            if np.dot(func(x + t_k * d, mode = 1, counter = cont)[0], d) >= sigma_C * np.dot(grad_x, d):
                return t_k
            else:
                L_k = t_k
        else:
            R_k = t_k

        if R_k == 0:
            t_k = extrapolation(t_k)
        else:
            t_k = interpolation(L_k, R_k)
        # k += 1

def grad_method(func, x0, t = 0.0001, iter_limit = 100, tol = 1e-05, line_search = 2):
    ''' Steepest gradient descent method for unconstrained non-linear optimisation '''
    x, theta, sigma_A, sigma_C = x0, 0.5, 0.4, 0.8
    current_f, grad_f = None, None
    for i in range(0, iter_limit):
        grad_f, _ = func(x, mode = 1, counter = cont)
        current_f, _ = func(x, mode = 0, counter = cont)

        # print ' -----------------------------'
        # print 'Iteration #', i + 1, 'w/ f(x) =', current_f
        # print 'x_bar =', x

        if np.linalg.norm(grad_f, np.inf) < tol:
            break

        direction = - grad_f

        if line_search == 0:
            t_k = cauchy_LS(func, x)
        elif line_search == 1:
            t_k = armijo_LS(func, x, direction, t, theta, sigma_A)
        elif line_search == 2:
            t_k = wolfe_LS(func, x, direction, t, sigma_A, sigma_C)
        elif line_search == 3:
            t_k = goldstein_LS(func, x, direction, t, sigma_A, sigma_C)

        # print 't_k =', t_k

        x += t_k * direction

    return x, current_f, grad_f

def conjugate_grad_method(func, x0, t = 0.0001, iter_limit = 100, tol = 1e-05):
    ''' Conjugate gradient descent method for unconstrained non-linear optimisation '''
    x, beta_k, theta, sigma_A, sigma_C = x0, 0.0, 0.5, 0.4, 0.8
    current_f, grad_f, previous_grad, direction = None, None, None, None
    for i in range(0, iter_limit):
        grad_f, _ = func(x, mode = 1, counter = cont)
        current_f, _ = func(x, mode = 0, counter = cont)

        # print ' -----------------------------'
        # print 'Iteration #', i + 1, 'w/ f(x) =', current_f
        # print 'x_bar =', x

        if np.linalg.norm(grad_f, np.inf) < tol:
            break

        if i == 0:
            direction = - grad_f
        else:
            beta_k = (np.linalg.norm(grad_f) ** 2) / (np.linalg.norm(previous_grad) ** 2)
            direction = - grad_f + beta_k * direction

        # Calculating cauchy exact stepsize for a quadratic function
        t_k = cauchy_LS2(func, grad_f, direction)

        # print 't_k =', t_k
        previous_grad = grad_f

        x += t_k * direction

    return x, current_f, grad_f

if __name__ == '__main__':
    ''' Main statements '''
    # x_bar = np.ones(2)
    x_bar = np.ones(4)
    # x_bar = np.random.rand(4)
    # x_bar = np.array([-0.1, -0.2])
    initial_t = 1 # = {1, 5} (for armijo's)
    iter_limit = 100
    tol = 1e-05

    # res, obj, grad = grad_method(sugar, x_bar, initial_t, iter_limit, tol, 2)
    # res, obj, grad = grad_method(power, x_bar, initial_t, iter_limit, tol, 0)
    res, obj, grad = conjugate_grad_method(power, x_bar, initial_t, iter_limit, tol)

    print '\n ---------- RESULT: ----------'
    print 'x* =', res, '\nw/ f(x*)=', obj
    print 'f\'(x*)=', grad
    print 'Number of evaluations:', cont
    print ' -----------------------------'
