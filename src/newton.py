import numpy as np
from testF import power, sugar

cont = {'fun':0, 'grad':0} # counter of function and gradient evaluations

def wolfe_LS(func, x, d, t0, sigma_A, sigma_C):
    ''' Wolfe Line Search stepsize method for descent optimisation methods '''

    def extrapolation(R_k, rho = 2):
        return rho * R_k

    def interpolation(L_k, R_k, theta = 0.5):
        return theta * L_k + (1 - theta) * R_k

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

# @profile
def newton_method(func, x0, iter_limit = 100, tol = 1e-05):
    ''' Newton method for unconstrained non-linear optimisation using inverse '''
    x = x0
    current_f, grad_f, hessian_f = None, None, None
    for i in range(0, iter_limit):
        grad_f, _ = func(x, mode = 1, counter = cont)
        current_f, _ = func(x, mode = 0, counter = cont)

        # print 'Iteration #', i + 1, 'w/ f(x) =', current_f
        # print 'x_bar = ', x
        if np.linalg.norm(grad_f, np.inf) < tol:
            break

        hessian_f, _ = func(x, mode = 3, counter = cont)
        hessian_inv = np.linalg.inv(hessian_f)
        x -= np.dot(hessian_inv, grad_f)

    return x, current_f

# @profile
def newton_method2(func, x0, iter_limit = 100, tol = 1e-05):
    ''' Newton method for unconstrained non-linear optimisation solving a linear system '''
    x = x0
    current_f, grad_f, hessian_f = None, None, None
    for i in range(0, iter_limit):
        grad_f, _ = func(x, mode = 1, counter = cont)
        current_f, _ = func(x, mode = 0, counter = cont)

        print 'Iteration #', i + 1, 'w/ f(x) =', current_f
        print 'x_bar = ', x
        if np.linalg.norm(grad_f, np.inf) < tol:
            break

        hessian_f, _ = func(x, mode = 3, counter = cont)
        x -= np.linalg.solve(hessian_f, grad_f)

    return x, current_f

def globalized_newton(func, x0, t = 1, iter_limit = 100, tol = 1e-05):
    ''' Globalized Newton method for unconstrained non-linear optimisation '''
    x, sigma_A, sigma_C = x0, 0.4, 0.8
    current_f, grad_f, hessian_f = None, None, None
    for i in range(0, iter_limit):
        grad_f, _ = func(x, mode = 1, counter = cont)
        current_f, _ = func(x, mode = 0, counter = cont)

        print 'Iteration #', i + 1, 'w/ f(x) =', current_f
        print 'x_bar = ', x
        if np.linalg.norm(grad_f, np.inf) < tol:
            break

        hessian_f, _ = func(x, mode = 3, counter = cont)
        direction, t_k = None, 1
        try:
            direction = np.linalg.solve(hessian_f, grad_f)
        except np.linalg.linalg.LinAlgError:
            direction = - grad_f
            t_k = wolfe_LS(func, x, direction, t, sigma_A, sigma_C)
        x -= t_k * direction

    return x, current_f, grad_f

def quasi_newton(func, x0, t = 1, iter_limit = 100, tol = 1e-05):
    ''' Quasi-Newton method for unconstrained non-linear optimisation '''

    def dfp_matrix(W_k, y_k, s_k):
        aux = np.dot(W_k, y_k)
        p1 = (s_k * s_k.reshape(len(s_k), 1)) / np.dot(y_k, s_k)
        p2 = aux * aux.reshape(len(aux), 1) / np.dot(y_k, aux)
        B_k = p1 - p2
        return W_k + B_k

    def bfgs_matrix(W_k, y_k, s_k):
        aux1 = np.dot(W_k, y_k)
        aux2 = np.dot(y_k, s_k)
        p1 = ((s_k * aux1.reshape(len(aux1), 1)) + (aux1 * s_k.reshape(len(s_k), 1))) / aux2
        p2 = (s_k * s_k.reshape(len(s_k), 1)) / aux2
        p3 = np.dot(y_k, aux1) / aux2
        return W_k - p1 + p2 + (p3 * p2)

    x, W_k, y_k, s_k, sigma_A, sigma_C = x0, np.eye(len(x0)), None, None, 0.4, 0.6
    current_f, _ = func(x, mode = 0, counter = cont)
    grad_f, _ = func(x, mode = 1, counter = cont)
    for i in range(0, iter_limit):
        # print ' -----------------------------'
        # print 'Iteration #', i + 1, 'w/ f(x) =', current_f
        # print 'x_bar =', x
        # print 'W_k =', W_k
        # print 'eigenvalues = ', np.linalg.eigvals(W_k)
        if np.linalg.norm(grad_f, np.inf) < tol:
            break

        direction = - np.dot(W_k, grad_f)
        t_k = wolfe_LS(func, x, direction, t, sigma_A, sigma_C)

        s_k = - x
        y_k = - grad_f

        x += t_k * direction

        current_f, _ = func(x, mode = 0, counter = cont)
        grad_f, _ = func(x, mode = 1, counter = cont)

        s_k += x
        y_k += grad_f
        # W_k = dfp_matrix(W_k, y_k, s_k)
        W_k = bfgs_matrix(W_k, y_k, s_k)

    return x, current_f, grad_f

if __name__ == '__main__':
    ''' Main statements '''
    # x_bar = np.ones(2)
    x_bar = np.random.rand(20)
    # x_bar = np.array([-0.1, -0.2]) # initial point for sugar function
    initial_t = 1 # = {1, 5} (for armijo's)
    iter_limit = 100
    tol = 1e-05

    # res, obj, grad = newton_method(power, x_bar)
    # res, obj, grad = newton_method2(power, x_bar)
    # res, obj, grad = globalized_newton(power, x_bar, initial_t, iter_limit, tol)
    res, obj, grad = quasi_newton(power, x_bar, initial_t, iter_limit, tol)

    print '\n ---------- RESULT: ----------'
    print 'x* =', res, '\nw/ f(x*)=', obj
    print 'f\'(x*)=', grad
    print 'Number of evaluations:', cont
    print ' -----------------------------'
