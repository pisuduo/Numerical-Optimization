#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 19 20:12:45 2017

@author: maziyang
"""

###Homework 1
import numpy as np
import matplotlib.pyplot as plt

###Part 1

###(a)Unif[0,1]
unisample=np.random.uniform(0,1,size=1000)
EMuni=0.5
EVuni=1/12.0
np.mean(unisample)
np.var(unisample)
print("the sample mean and expected value of mean are " + str(np.mean(unisample))+" and "+str(EMuni)) 
print("the sample variance and expected value of variance are " + str(np.var(unisample))+" and "+str(EVuni)) 


fig,ax=plt.subplots()
ax.hist(unisample,20,normed=1,histtype='bar',edgecolor='black',alpha=0.8)  
plt.title("Histogram of Uniform Random Samples")
plt.xlabel("Value")
plt.ylabel("Density")
plt.savefig("Histunisample.png") 

###(b)
expsample=np.random.exponential(scale=1.5,size=1000)
EMexp=1.5
EVexp=1.5**2
np.mean(expsample)
np.var(expsample)
print("the sample mean and expected value of mean are " + str(np.mean(expsample))+" and "+str(EMexp)) 
print("the sample variance and expected value of variance are " + str(np.var(expsample))+" and "+str(EVexp))

fig,ax=plt.subplots()
ax.hist(expsample,20,normed=1,histtype='bar',edgecolor='black',alpha=0.8)  
plt.title("Histogram of Exponential Random Samples")
plt.xlabel("Value")
plt.ylabel("Density")
plt.savefig("Histexpsample.png")



###(c)
cauchysample=np.random.standard_cauchy(size=1000)
np.mean(cauchysample)
np.var(cauchysample)
print("the sample mean and expected value of mean are " + str(np.mean(cauchysample))+" and "+"undefined") 
print("the sample variance and expected value of variance are " + str(np.var(cauchysample))+" and "+"undefined")

fig,ax=plt.subplots()
ax.hist(cauchysample,20,normed=1,histtype='bar',edgecolor='black',alpha=0.8) 
plt.title("Histogram of Standard Cauchy Random Samples")
plt.xlabel("Value")
plt.ylabel("Density")
plt.savefig("Histcauchysample.png")


###(d)
normalsample=np.random.normal(loc=1,scale=1,size=1000)
EMnorm=1
EVnorm=1
np.mean(normalsample)
np.var(normalsample)
print("the sample mean and expected value of mean are " + str(np.mean(normalsample))+" and "+str(EMnorm)) 
print("the sample variance and expected value of variance are " + str(np.var(normalsample))+" and "+str(EVnorm))

fig,ax=plt.subplots()
ax.hist(normalsample,20,normed=1,histtype='bar',edgecolor='black',alpha=0.8) 
plt.title("Histogram of Normal Random Samples")
plt.xlabel("Value")
plt.ylabel("Density")
plt.savefig("Histnormalsample.png")



###(e)
chisample=np.random.chisquare(5,size=1000)
EMchi=5
EVchi=2*5
np.mean(chisample)
np.var(chisample)
print("the sample mean and expected value of mean are " + str(np.mean(chisample))+" and "+str(EMchi)) 
print("the sample variance and expected value of variance are " + str(np.var(chisample))+" and "+str(EVchi))

fig,ax=plt.subplots()
ax.hist(chisample,20,normed=1,histtype='bar',edgecolor='black',alpha=0.8)
plt.title("Histogram of Chisquare Random Samples")
plt.xlabel("Value")
plt.ylabel("Density")
plt.savefig("Histchisample.png")


###(f)
binomialsample=np.random.binomial(10,0.6,size=1000)
EMbin=10*0.6
EVbin=10*0.6*0.4
np.mean(binomialsample)
np.var(binomialsample)
print("the sample mean and expected value of mean are " + str(np.mean(binomialsample))+" and "+str(EMbin)) 
print("the sample variance and expected value of variance are " + str(np.var(binomialsample))+" and "+str(EVbin))

fig,ax=plt.subplots()
ax.hist(binomialsample,normed=1,histtype='bar',edgecolor='black',alpha=0.8)
plt.title("Histogram of Binomial Random Samples")
plt.xlabel("Value")
plt.ylabel("Density")
plt.savefig("Histbinomialsample.png")


###(g)
geosample=np.random.geometric(p=0.35,size=1000)
EMgeo=1/0.35
EVgeo=(1-0.35)/(0.35**2)
np.mean(geosample)
np.var(geosample)
print("the sample mean and expected value of mean are " + str(np.mean(geosample))+" and "+str(EMgeo)) 
print("the sample variance and expected value of variance are " + str(np.var(geosample))+" and "+str(EVgeo))

fig,ax=plt.subplots()
ax.hist(geosample,20,normed=1,histtype='bar',edgecolor='black',alpha=0.8)
plt.title("Histogram of Geometric Random Samples")
plt.xlabel("Value")
plt.ylabel("Density")
plt.savefig("Histgeosample.png")



###(h)
gammasample=np.random.gamma(shape=3.5,scale=5,size=1000) 
EMgamma=3.5*5
EVgamma=3.5*(5**2)
np.mean(gammasample)
np.var(gammasample)
print("the sample mean and expected value of mean are " + str(np.mean(gammasample))+" and "+str(EMgamma)) 
print("the sample variance and expected value of variance are " + str(np.var(gammasample))+" and "+str(EVgamma))

fig,ax=plt.subplots()
ax.hist(gammasample,20,normed=1,histtype='bar',edgecolor='black',alpha=0.8)
plt.title("Histogram of Gamma Random Samples")
plt.xlabel("Value")
plt.ylabel("Density")
plt.savefig("Histgammasample.png")

### Part 2
###a
def unirg(n,inival):
    import numpy as np
    a=7**5
    m=2**32-1
    I=np.zeros(n+1)
    I[0]=inival
    for i in range(n):
        I[i+1]=((a*I[i])%m)
    return(I[-n:]/m)

### an example of running the function, 
unirg(20,1.0) ##choosing a sample size of 20,a intial value(seed) of 1.0
np.mean(unirg(1000,1.0))
np.var(unirg(1000,1.0))

###b     
### by default, theta is the scale parameter
def exprg(theta,n):
    import numpy as np
    e=-(np.log(1-np.random.uniform(0,1,size=n)))*theta
    return(e)
### an example of running the function
exsam=exprg(1.5,1000) ##generate 1000 samples from exp(1.5)
np.mean(exsam)
np.var(exsam)


###c standard cauchy random number generator
def caurg(n):
    import numpy as np
    import math as m
    x=np.tan(m.pi*(np.random.uniform(0,1,size=n)-0.5))
    return(x)
### an example of running the function
caurg(10)
np.mean(caurg(10))

### d normal distribution
def normrg(n,mu,sigma):
    import numpy as np
    import math as m
    if n%2==0:
        u=np.random.uniform(0,1,size=n)
        Z=np.zeros(n)
        X=np.zeros(n)
        for i in range(0,n,2):
            Z[i]=m.sqrt(-2.0*m.log(u[i]))*m.cos(2*m.pi*u[i+1])
            Z[i+1]=m.sqrt(-2.0*m.log(u[i]))*m.sin(2*m.pi*u[i+1])
            X[i]=sigma*Z[i]+mu
            X[i+1]=sigma*Z[i+1]+mu
        return(X)
    else:
        u=np.random.uniform(0,1,size=n+1)
        Z=np.zeros(n+1)
        X=np.zeros(n+1)
        for i in range(0,n+1,2):
            Z[i]=m.sqrt(-2.0*m.log(u[i]))*m.cos(2*m.pi*u[i+1])
            Z[i+1]=m.sqrt(-2.0*m.log(u[i]))*m.sin(2*m.pi*u[i+1])
            X[i]=sigma*Z[i]+mu
            X[i+1]=sigma*Z[i+1]+mu
        return(X[-n:])

### an example of running the function    
normrg(10,1,1)
np.mean(normrg(1000,100,2))
np.var(normrg(1000,100,2))
plt.hist(normrg(100000,1,10))


### e chisquare random numbers
def chisqrg(n,df):
    import numpy as np
    import math as m
    x=np.zeros(n)
    for i in range(n):
        norm=normrg(df,0,1)  ### using the normal random number generator in d
        x[i]=np.sum(norm**2)
    return(x)

chisqrg(10,5)
np.mean(chisqrg(1000,5))
np.var(chisqrg(1000,5))

### f 
def binrg(N,p,n):
    import numpy as np
    x=np.zeros(N)
    y=np.zeros(n)
    for i in range(n):
        for j in range(N):
            u=np.random.uniform(0,1,size=1)
            if u<=p:
                x[j]=1
            else:
                x[j]=0
        y[i]=np.sum(x)
    return(y)

### an example of running the function
binrg(5,0.5,1000)
np.mean(binrg(5,0.5,1000))
np.var(binrg(5,0.5,1000))

### g
def georg(p,n):
    import numpy as np
    import math as m
    lamda=-m.log(1-p)
    x=np.zeros(n)
    y=np.zeros(n)
    for i in range(n):
        x[i]=exprg(1/lamda,1)
        y[i]=int(x[i])+1
    return(y)

### an example of running the function
georg(0.5,1000)

np.mean(georg(0.6,1000))
np.var(georg(0.6,1000))
        
     
        
        
### part 3
import math as m
import numpy as np
import scipy.stats as ss
import matplotlib.pyplot as plt


### choose uniform distribution as g(x),g(x)=1

### the constant c is calculated as 
c=6*0.5*0.5

def betarejsam(n):###n is the number of sample to be generated
    x=np.zeros(n)
    for i in range(n):
        flag=0
        while flag==0:
            u=np.random.uniform(0,1,size=1)
            y=np.random.uniform(0,1,size=1)
            if u<=4*y*(1-y):
                x[i]=y
                flag=1
    return(x)
        
##example of running the function
betasam=betarejsam(1000)  ##generate 1000 samples form beta(2,2)
np.mean(betarejsam(1000)) ## the mean is close to its expected value of 0.5
np.var(betarejsam(1000))  ## the variance is close to its expected value of 0.05

##plot the data
fig,ax=plt.subplots()
ax.hist(betasam,20,normed=1,histtype='bar',edgecolor='black',facecolor='pink',alpha=0.8) 
plt.title("Histogram of Beta(2,2) Random Samples")
plt.xlabel("Value")
plt.ylabel("Density")
plt.savefig("Histbetasample.png")
 

###Part 4
def gammawhit(a,b,n): ## a is the shape parameter, b is the scale parameter, n is sample size
    import math as m
    import numpy as np
    aint=int(a)
    p=a-aint
    x=np.zeros(n)
    x1=np.zeros(n)
    x2=np.zeros(n)
    x3=np.zeros(n)
    for i in range(n):
        sum=0
        flag=0
        for j in range(aint):
            y=np.random.exponential(b)
            sum=sum+y
        x1[i]=sum
        while flag==0:
            u1=np.random.uniform(0,1,size=1)
            u2=np.random.uniform(0,1,size=1)
            s1=u1**(1/p)
            s2=u2**(1/(1-p))
            if (s1+s2)<=1:
                x2[i]=s1/(s1+s2)
                flag=1
        u3=np.random.uniform(0,1,size=1)
        x3[i]=-b*x2[i]*m.log(u3)
        x[i]=x1[i]+x3[i]
    return(x)

##example of running the function
gamsam=gammawhit(1.5,2,1000)   ## generate 1000 samples from gamma(1.5,2)   
np.mean(gamsam) ##sample mean is close to its expected value of 1.5*2=3
np.var(gamsam)  ##sample variance is close to its expected value of 1.5*2*2=6

##plot the data
fig,ax=plt.subplots()
ax.hist(gamsam,20,normed=1,histtype='bar',edgecolor='black',facecolor='pink',alpha=0.8) 
plt.title("Histogram of Gamma(1.5,2) Random Samples")
plt.xlabel("Value")
plt.ylabel("Density")
plt.savefig("Histgamsam.png")
   
 ###the shape is close to the expected shape            


       
        
    
        

 












