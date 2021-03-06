#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  8 14:48:56 2017

@author: maziyang
"""

import numpy as np
import numpy.random as npr
import numpy.matlib as ml

## make sequence
def MakeSeq(begin, end, by):
    A=[]
   
    i=0
    while (begin+by*i)<end:
        A.append(begin+by*i)
        i=i+1
    return(A)

n=20
xvals=MakeSeq(-10,10,1)
yvals=np.mat([4+3*xvals[i]+xvals[i]**2+npr.normal(0,15,1) for i in range(len(xvals))])

X=ml.empty([n,3])
for i in range(n):
    X[i,0]=1
    X[i,1]=xvals[i]
    X[i,2]=xvals[i]**2
    
    
    
A=X.T*X
V=np.matrix(range(3))
Y=X.T*yvals  ## Y vector


###############################################################################
###############################################################################
#### Problem 1

#### define a function to swap the rows
def swap(A,a,b): ## change ath and bth rows of A
    n=A.shape[1] #number of columns for 2d array A
    for j in range(n):
        temp=A[b-1,j]
        A[b-1,j]=A[a-1,j]
        A[a-1,j]=temp
    return(A)  ## return the swapped array


### define the function to do LU decomposition
def LUdecomp(A):
    A=A.astype(float)  ## make A float
    m=A.shape[0] ## row of A matrix
    L=np.eye(m)  ## intialize L
    U=np.eye(m)  ## intialize U
    B=A          ## B will store the updating matrix
    #z=[]         ## z will store the information of permutation
    z=[1,2,3]
    for k in range(m):
        for j in range(k,m):
            U[k,j]=A[k,j]-sum([L[k,n]*U[n,j] for n in range(k)])
            B[k,j]=U[k,j]   ## update matrix
        for i in range(k+1,m):
            L[i,k]=(A[i,k]-sum([L[i,n]*U[n,k] for n in range(k)]))
            B[i,k]=L[i,k]
        if k+1<m:    ## when k is not the last row, check for swap condition
            if abs(U[k,k])>max(abs(np.array([L[x,k] for x in range(k+1,m)]))):
                imax=k  
            else:
                imax=np.argmax(abs(np.array([L[x,k] for x in range(k+1,m)])))+k+1
        else:
            imax=k   ## when k achieves the last row, don't swap
        if imax!=k:  ## update z
            temp=z[imax]
            z[imax]=z[k]
            z[k]=temp
        #z.append([imax+1,k+1]) ## z is a list containing the swap information. 
        ## swap vector z[0] is the swap information for first step, 
        ##the two elements in z[0] represents the rows to be swapped
        ##z[1] is the swap information for second step..
        
        swap(B,imax+1,k+1) ## swap the k+1 th row and imax+1 th row, if imax=k, don't swap
        for i in range(k+1,m):  ## update L after swapping
            L[i,k]=1.0*B[i,k]/B[k,k]
            B[i,k]=L[i,k]
        U=B ## updating matrix
        L=B ## updating matrix
    A=B ## modify A
    return(A,z)  ## return the modified A and the swapping information


## run the function and give the results
print("LU decomposition from my code:")
print(LUdecomp(A))

## the cooresponding L and U are
L=np.array([[1,0,0],[-1.4925*10**(-2),1,0],[2.985*10**(-2),3.0303*10**(-2),1]])
U=np.array([[670,-1000,4.0666*(10**(4))],[0,6.5507*10**(2),-3.9305*10**(2)],[0,0,-5.32*10**(2)]])
np.matmul(L,U)


###############################################################################
###############################################################################
### Problem 2

def LUsub(B,Y,z):
    ## input B is the modified A from part 1
    ## input z is a list containging swaping information
    Y=Y.astype(float) #make Y float
    Y=np.array(Y)     ##make Y an arary
    B=B.astype(float)  #make B float
    m=B.shape[0]     ## rows of B
    L=np.eye(m)      ## initialize L
    U=np.eye(m)      ## initialize U
    Z=np.zeros(m)    ## initialize Z, which will be the solution for LZ=Y
    X=np.zeros(m)    ## initialize X, which will be the solution for UX=Z
    ##since A has been swapped, Y need to have the cooresponding swaps
    for i in range(m):
        swap(Y,z[i][0],z[i][1]) ## return the swapped Y
    ## calculate L and U from B
    for i in range(1,m): ##L
        for j in range(i):
            L[i,j]=B[i,j]
    for k in range(m):   ##U
        for q in range(0,k+1):
            U[q,k]=B[q,k]
    ## sloving Z from LZ=Y by forward substitution
    for k in range(m):
        Z[k]=(Y[k]-sum([Z[n]*L[k,n] for n in range(k)]))/L[k,k]
    ## solving X from UX=Z by backward substitution
    for j in range(m):
        X[m-1-j]=(Z[m-1-j]-sum([X[n]*U[m-1-j,n] for n in range(m-j,m)]))/U[m-1-j,m-1-j]
    Y=X ## replace Y by X
    return(Y) ## return the solution for the equation


B=LUdecomp(A)[0] ## modified A from LU decom
LUdecomp(A)[1]
## I made a list to better illustrate how the premutation works, and I use this for my input in LU sub function
z=[[1,3],[2,2],[3,3]]
## swap vector z[0] is the swap information for first step, 
#the two elements in z[0] represents the rows to be swapped
##z[1] is the swap information for second step..
print("Parameter estimates by LU decompsition from my code:")
print(LUsub(B,Y,z))

