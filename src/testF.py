import numpy as np

def diagonal7(x, mode=0, counter={}):
    if mode==0:
        if counter:
            counter['fun'] += 1
        return sum(np.exp(x) - 2 * x - x ** 2), 0
    elif mode==1:
        if counter:
            counter['grad']+=1
        return np.exp(x) - 2 * x - 2, 0
    elif mode==2:
        if counter:
            counter['fun'] += 1
            counter['der']+=1
        return sum(np.exp(x) - 2 * x - x ** 2), np.exp(x) - 2 * x - 2, 0

def power(x, mode = 0,counter = {}):
    '''A blackbox method for evaluating the Power function (CUTE) and its Jacobian and Hessian'''
    aux = np.arange(1,len(x) + 1)
    if mode == 0: # Return f(x), status
        if counter:
            counter['fun'] += 1
        return sum((aux * x) ** 2), 0
    elif mode == 1: # Return f'(x), status
        if counter:
            counter['grad'] += 1
        return 2 * ((aux ** 2) * x), 0
    elif mode == 2: # Return f(x), f'(x), status
        if counter:
            counter['fun'] += 1
            counter['grad'] += 1
        return sum((aux * x) ** 2), 2 * ((aux ** 2) * x), 0
    elif mode == 3:
        return np.eye(len(x)) * (2 * (aux ** 2)), 0

def sugar(x, mode = 0,counter = {}):
    '''A blackbox method for evaluating the Sugar function and its Jacobian and Hessian'''
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

def raydan1(x, mode=0, counter={}):
    aux = np.arange(1, len(x) + 1) / 10
    if mode == 0:
        if counter:
            counter['fun'] += 1
        return sum(aux * (np.exp(x) - x)), 0
    elif mode == 1:
        if counter:
            counter['grad'] += 1
        return aux * (np.exp(x) - x), 0
    elif mode == 2:
        if counter:
            counter['fun'] += 1
            counter['grad'] += 1
        return sum(aux * (np.exp(x) - x)), aux * (np.exp(x) - x), 0
