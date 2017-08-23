import numpy as np

cont = {'fun':0, 'grad':0} # counter of function and gradient evaluations

def read_data(folder):
    # Reading lines from files
    config_lines = [line.rstrip('\n') for line in open(folder + 'config.csv')]
    k_lines = [line.rstrip('\n') for line in open(folder + 'k.csv')]
    mass_lines = [line.rstrip('\n') for line in open(folder + 'mass.csv')]

    # Manipulating the read data
    config_lines = config_lines[1:]
    a = [float(config_lines[0].split(',')[0]), float(config_lines[1].split(',')[0])]
    b = [float(config_lines[0].split(',')[1]), float(config_lines[1].split(',')[1])]
    g = float(config_lines[0].split(',')[2])
    k_lines = [float(line) for line in k_lines]
    mass_lines = [float(line) for line in mass_lines]

    return np.array(a), np.array(b), g, np.array(k_lines), np.array(mass_lines)

folder = '../instances/examples3_3-3_4/3_4/case3/'
a, b, g, K, mass = read_data(folder)

def chain(x, mode = 0,counter = {}):
    '''
    A blackbox method for Example 3.4
    '''
    n = len(mass)
    xx = x.reshape(2, n)
    if mode == 0:
        if counter:
            counter['fun'] += 1

        diff1 = [xx[0, 0] - a[0]]
        diff2 = [xx[1, 0] - a[1]]
        for i in range(n - 1):
            diff1.append(xx[0, i + 1] - xx[0, i])
            diff2.append(xx[1, i + 1] - xx[1, i])
        diff1.append(b[0] - xx[0, n - 1])
        diff2.append(b[1] - xx[1, n - 1])

        diff = np.array(diff1 + diff2).reshape(2, n + 1)
        diff = np.array([np.linalg.norm(diff[:, i]) for i in range(n + 1)])
        # print diff

        return np.dot(diff ** 2, K / 2) + g * np.dot(mass, xx[1, :n]), 0
    elif mode == 1:
        if counter:
            counter['grad'] += 1

        aux1, aux2, value = [], [], None
        for i in range(n):
            if (i - 1) < 0:
                # aux1.append(2 * K[i + 1] * xx[0, i] - K[i] * a[0] - K[i + 2] * xx[0, i + 1])
                # aux2.append(2 * K[i + 1] * xx[1, i] - K[i] * a[1] - K[i + 2] * xx[1, i + 1] + g * mass[i])
                # value = K[i] * (2 * xx[:, i] - a - xx[:, i + 1])
                aux1.append(K[i] * (2 * xx[0, i] - a[0] - xx[0, i + 1]))
                aux2.append(K[i] * (2 * xx[1, i] - a[1] - xx[1, i + 1]) + g * mass[i])
            elif (i + 1) >= n:
                # value = K[i] * (2 * xx[:, i] - xx[:, i - 1] - b)
                aux1.append(K[i] * (2 * xx[0, i] - xx[0, i - 1] - b[0]))
                aux2.append(K[i] * (2 * xx[1, i] - xx[1, i - 1] - b[1]) + g * mass[i])
            else:
                # value = K[i] * (2 * xx[:, i] - xx[:, i - 1] - xx[:, i + 1])
                aux1.append(K[i] * (2 * xx[0, i] - xx[0, i - 1] - xx[0, i + 1]))
                aux2.append(K[i] * (2 * xx[1, i] - xx[1, i - 1] - xx[1, i + 1]) + g * mass[i])
            # aux.append(value)
            # mge.append(mass[i] * g * np.array([0, 1]))

        # return np.array(np.array(aux) + np.array(mge)).reshape(2 * n), 0
        return np.array(aux1 + aux2), 0
        # aux = [(K[i] * (2 * xx[:, i] - xx[:, i - 1] - xx[:, i + 1])) from i in range(n)]
        # return K * (2 * xx[:, 1:] - xx[])
    elif mode == 2:
        if counter:
            counter['fun'] += 1
            counter['der'] += 1

from gradient import grad_method, conjugate_grad_method
from newton import quasi_newton

if __name__ == '__main__':
    ''' Main statements '''
    x = np.random.rand(len(mass) * 2)
    # x = np.array([0.85454128, 0.33411715, 0.53942535, 0.21662987, 0.63774436, 0.79173458, 0.21410974, 0.15384516, 0.76524993, 0.83061825, 0.85518546, 0.57445061, 0.44503847, 0.47772461, 0.34943261, 0.6835797, 0.69350642, 0.04454588, 0.93143294, 0.11527991])
    # x = np.ones(len(mass) * 2)
    initial_t = 1 # = {1, 5} (for armijo's)
    iter_limit = 10000
    tol = 1e-05

    # res, obj, grad = grad_method(chain, x, initial_t, iter_limit, tol, 2)
    res, obj, grad = conjugate_grad_method(chain, x, initial_t, iter_limit, tol)
    # res, obj, grad = quasi_newton(chain, x, initial_t, iter_limit, tol)

    print '\n\n ---------- RESULT: ----------'
    print 'x* =', res, '\nw/ f(x*)=', obj
    print 'f\'(x*) =', grad
    # print 'Number of evaluations:', cont
    print ' -----------------------------'
