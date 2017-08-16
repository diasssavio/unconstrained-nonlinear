import matplotlib.pyplot as plt
import numpy as np
# import mathp.opt.tools as tt

cont = {'fun':0, 'grad':0} # counter of function and gradient evaluations
def sugar(x, mode = 0,counter = {}):
    '''A blackbox method for evaluating functions and its derivatives'''
    if mode == 0: # Return f(x), status
        if counter:
            counter['fun'] += 1
        return x.sum() * np.exp(-6.0 * np.dot(x,x)), 0
    elif mode == 1: # Return f'(x), status
        if counter:
            counter['grad'] += 1
        return np.exp(-6.0 * np.dot(x,x)) * (np.ones_like(x) - 12 * x.sum() * x), 0
    elif mode == 2: # Return f(x), f'(x), status
        if counter:
            counter['fun'] += 1
            counter['grad'] += 1
        return x.sum() * np.exp(-6.0 * np.dot(x, x)), np.exp(-6.0 * np.dot(x,x)) * (np.ones_like(x) - 12 * x.sum() * x), 0
    elif mode == 3:
        aux = -12 * np.exp(-6.0 * np.dot(x,x)) * x
        aux1 = np.ones_like(x) - 12 * x.sum() * x
        aux2 = -12 * (np.dot(x.reshape(len(x),1), np.ones_like(x).reshape(1,len(x))) + x.sum() * np.eye(len(x)))
        return np.dot(aux1.reshape(len(aux1), 1), aux.reshape(1,len(aux))) + np.exp(-6.0 * np.dot(x,x)) * aux2, 0

def rosenbrock(x, mode = 0,counter = {}):
    '''A blackbox method for evaluating the Rosenbrock function and its derivatives'''
    if mode == 0: # Return f(x), status
        if counter:
            counter['fun'] += 1
        return (1 - x[0]) ** 2 + 100 * (x[1] - x[0] ** 2) ** 2, 0
    elif mode == 1: # Return f'(x), status
        if counter:
            counter['grad'] += 1
        aux1 = 200 * (x[1] - (x[0] ** 2))
        aux2 = 4 * x[0] * (101 * (x[0] ** 2) - 100 * x[1] - 1)
        return np.array([aux1, aux2]), 0
    elif mode == 2: # Return f(x), f'(x), status
        if counter:
            counter['fun'] += 1
            counter['grad'] += 1
        return (1 - x[0]) ** 2 + 100 * (x[1] - x[0] ** 2) ** 2, np.array([200 * (x[1] - x[0] ** 2), 4 * x[0] * (101 * (x[0] ** 2) - 100 * x[1] - 1)]), 0
    # elif mode == 3:
    #     aux = -12 * np.exp(-6.0 * np.dot(x,x)) * x
    #     aux1 = np.ones_like(x) - 12 * x.sum() * x
    #     aux2 = -12 * (np.dot(x.reshape(len(x),1), np.ones_like(x).reshape(1,len(x))) + x.sum() * np.eye(len(x)))
    #     return np.dot(aux1.reshape(len(aux1), 1), aux.reshape(1,len(aux))) + np.exp(-6.0 * np.dot(x,x)) * aux2, 0

def test_sugar():
    x = np.zeros(2)
    y = sugar(x, mode=0, counter=cont)
    print(y)
    print(cont)
    y= sugar(x, mode=1, counter=cont)
    print(y)
    print(cont)
    y= sugar(x, mode=2, counter=cont)
    print(y)
    print(cont)
    y= sugar(x, mode=3, counter=cont)
    print(y)
    print(cont)

def test_rosenbrock():
    x = np.zeros(2)
    # x = np.array([10,5])
    y = rosenbrock(x, mode=2)
    print(x)
    print(y)

def checkBB(func, x, h):
    g, status = func(x, mode = 1)
    g_num = [0.0, 0.0]
    _x = [x[0] + h, x[1]], [x[0], x[1] + h]
    g_num = (func(_x[0])[0] - func(x)[0]) / h, (func(_x[1])[0] - func(x)[0]) / h
    erro = g - g_num
    erro = abs(erro[0]), abs(erro[1])
    if erro[0] > erro[1]:
        return erro[0]
    else:
        return erro[1]
x_bar = np.random.rand(2)
print sugar(x_bar, mode = 3)
# res = checkBB(rosenbrock, np.zeros(2), 0.00001)
# print res
# test_rosenbrock()
# test_sugar()
# tt.plot(sugar, x=[-0.5,0.5], y=[-0.5,0.5])
