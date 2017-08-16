import matplotlib.pyplot as plt
import numpy as np

cont = {'fun':0, 'grad':0} # counter of function and gradient evaluations
def power(x, mode = 0,counter = {}):
    '''A blackbox method for evaluating functions and its derivatives'''
    if mode == 0: # Return f(x), status
        if counter:
            counter['fun'] += 1
        value = 0.0
        for i in range(len(x)):
            value += ((i + 1) * x[i]) ** 2
        return value, 0
    elif mode == 1: # Return f'(x), status
        if counter:
            counter['grad'] += 1
        grad = []
        for i in range(len(x)):
            grad.append(2 * x[i] * ((i + 1) ** 2))
        return np.array(grad), 0
    elif mode == 2: # Return f(x), f'(x), status
        if counter:
            counter['fun'] += 1
            counter['grad'] += 1
        value = 0.0
        grad = []
        for i in range(len(x)):
            value += ((i + 1) * x[i]) ** 2
            grad.append(2 * x[i] * ((i + 1) ** 2))
        return value, np.array(grad), 0

def test_power(x):
    f_cost, grad, _ = power(x, mode=2, counter=cont)
    print 'f(x) = ', f_cost
    print 'f\'(x) = ', grad

def checkBB(func, x, h = 0.00001):
    g, _ = func(x, mode = 1, counter = cont)
    _x = []

    # Generating x based on definition, i.e. x + h
    for j in range(len(x)):
        aux = []
        for i in range(len(x)):
            if i == j:
                aux.append(x[i] + h)
            else:
                aux.append(x[i])
        _x.append(aux)
    # _x = [x[0] + h, x[1]], [x[0], x[1] + h]

    # Calculating numeric derivative of x
    g_num = []
    func_value, _ = func(x, mode = 0, counter = cont)
    for i in range(len(x)):
        g_num.append((func(_x[i], mode = 0, counter = cont)[0] - func_value) / h)
    # g_num = (func(_x[0])[0] - func(x)[0]) / h, (func(_x[1])[0] - func(x)[0]) / h
    erro = g - g_num
    erro = [abs(erro[i]) for i in range(len(x))]
    return max(erro)

x_bar = np.random.rand(4)
# x_bar = np.array([1,1,1,1])
# x_bar = np.array([0,0,0,0])
print 'x: ', x_bar
test_power(x_bar)
# res = checkBB(power, np.array([1, 1, 1, 1]))
res = checkBB(power, x_bar)
print 'Numeric derivative error: ', res
print 'Calculus count: ', cont
