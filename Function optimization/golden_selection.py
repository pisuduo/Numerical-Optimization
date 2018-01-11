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


### golden section search method
def goldsection(f,a,b,tol,maxit):
    n=1
    s=(1+m.sqrt(5))/2.0
    while n<=maxit:
        c=b-(b-a)/s
        d=a+(b-a)/s
        e=abs(c-d)
        if e<tol:
            print("iteration="+str(n-1)+",converge")
            print("minimum found is")
            break
        elif f(c)<f(d):
            b=d
            n=n+1
        else:
            a=c
            n=n+1
    if n>maxit:
        print("achieve maximum iteration times")
    return((b+a)/2)
    
## example function
def minf(x):
    return(x**2+2*x+4)

goldsection(minf,-2,0,10**(-6),30)      ## found the minimum position of minf, which is -1
