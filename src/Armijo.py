# -*- coding: utf-8 -*-
"""
Created on Mon Jul 10 14:40:41 2017

@author: AD
"""
import numpy as np
import os
os.getcwd()
os.chdir("E:/Fabio/COPPE-PPE/DSC/2 Periodo/PNL")
os.getcwd()

x=np.array([2,4])
e = 10**(-6)
cout={'fun':0,'grad':0}

#função POWER
def BBpower(x, mode = 0,counter = {}): #função POWER
    '''A blackbox method for evaluating functions and its derivatives'''
    aux = np.arange(1,len(x) + 1)
    if mode == 0: # Return f(x), status
        if counter:
            counter['fun'] += 1
            val= sum((aux * x) ** 2)
        return val, 0
    elif mode == 1: # Return f'(x), status
        if counter:
            counter['grad'] += 1
            val=2*((aux**2) * x)
        return val, 0
    elif mode == 2: # Return f(x), f'(x), status
        if counter:
            counter['fun'] += 1
            counter['grad'] += 1
            val=sum((aux * x) ** 2), 2*((aux**2) * x)
        return val, 0
 
#rosembrock function
def rosen(z, mode=0,contador={}):
    x,y=z
    rosenbroke=(1-x)**2+100*(y-x**2)**2
    derrosen=np.array([-2*(1-x)-400*(y-x**2)*x, 200*(y-x**2)])
    if mode==0:
        cout['fun']+=1
        return rosenbroke, 0
    elif mode==1:
        cout['grad']+=1
        return derrosen, 0
    elif mode== 2:
        cout['grad']+=1
        cout['fun']+=1
        return rosenbroke,derrosen , 0

    
A=0.8 #sigma A --> cte Armijo
B=0.5 #teta
def armijoLS(x, t, A, B, d, BB): #d=direção gradiente, x=x0, t =t0, BB=blackbox da função
    tlist=[t]
    k=0
    while(BB(x+t*d, 0, cout)[0] > (BB(x,0,cout)[0]+A*np.inner(BB(x,1,cout)[0],t*d))):
        k=k+1
        t=B*t
        tlist.append(t)
    x=x+t*d
    return [t, x, k, tlist]

"""
armijoLS(x, 5, A, B, d)
"""

def DescentMethod(x,e, BB, K, cout={}): #x=x0, e=erro, BB=blackbox da função, K=max de iteracoes, cout= evaluation
    xlist = [x]
    tlist=[]
    k=0
    rr=open('gradM_t.txt', 'w')
    while ((np.linalg.norm(BB(x,1,cout)[0], np.inf)) > e) and (k<K):
        d = -BB(x,1,cout)[0]
        x, t = armijoLS(x, 5, A, B,d, BB)[1],  armijoLS(x, 5, A, B, d, BB)[0]
        xlist.append(x)
        tlist.append(t)
        k += 1
        r=('\n k=%s --> t, x = %s, %s' %(k, t, x))
        rr.write(r)
    rr.close()
    return [x, BB(x,0,cout)[0], BB(x,1,cout)[0], cout, k, xlist, tlist]



resp = DescentMethod(x, e, rosen, 1000, cout)
print(resp[:-2])