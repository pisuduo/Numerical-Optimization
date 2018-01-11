#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 10:05:59 2017

@author: maziyang
"""

import numpy as np
import math as m
import pylab as pl
import matplotlib.pyplot as plt
import scipy as sp

### secant method
def secant(f,x0,x1,tol,maxit):
    n=1
    while n<=maxit:
        x2=x1-f(x1)*(x1-x0)/(f(x1)-f(x0))
        re=abs((x2-x1)/x2)
        if re<tol:
            print("iteration="+str(n-1)+", converge")
            print("root found is")
            break
        else:
            print("iteration="+str(n)+",xi-1="+str(x0)+",xi="+str(x1)+",xi+1="+str(x2)+",relative error="+str(re))
            n=n+1
            x0=x1
            x1=x2
    if n>maxit:
        print("achieve maximum iteration times")
    return(x2)

## example function
def exf(x):
    return(x**3-0.165*(x**2)+3.993*10**(-4))

secant(exf,0.02,0.05,10**(-6),20)  ## find a right root of 0.06238
'''
if we start at right locations for the other two roots, we will end up with the right roots.
'''
secant(exf,-0.01,0.02,10**(-6),20) ## find the second right root of -0.04374

secant(exf,0.10,0.12,10**(-6),20)  ## find the thrid right root of 0.14646
'''
Thus, the function works correctly on the example function
'''
