# -*- coding: utf-8 -*-
"""
Source code for the itmc part of the third synthetic example (Repeated evaluation of Case ii & iii, Figure 4 & 5) in
"Is My Model Flexible Enough? Information-Theoretic Model Check",
Andreas Svensson, Dave Zachariah, Thomas B. Schön arXiv:NNNN

Code by Andreas Svensson 2017
"""
import numpy as np
from scipy.stats import norm
import matplotlib.pylab as plt

class normdist():
    def __init__(self,m,sigma2):
        self.m = m
        self.sigma2 = sigma2
        
    def sample(self, N):
        return self.m + np.sqrt(self.sigma2)*norm.rvs(size=N)

class arm():
    def __init__(self,n,y0,sigma2):
        self.n = n # number of lags
        self.y0 = y0 # should be of size n
        self.sigma2 = sigma2
        
    def set_th(self,th):
        self.th = th # should be of size n
        
    def simulate(self, T):
        y = np.zeros(T)
        y[0:self.n] = self.y0
        for t in range(self.n,T):
            y[t] = np.sum(self.th*y[t-self.n:t]) + np.sqrt(self.sigma2)*norm.rvs()
        self.y = y
        
    def simulate_with_saturation(self, T):
        y = np.zeros(T)
        y[0:self.n] = self.y0
        for t in range(self.n,T):
            y[t] = np.max(np.array([np.sum(self.th*y[t-self.n:t]) + np.sqrt(self.sigma2)*norm.rvs(),-0.3]))
        self.y = y
        
    def Phi(self, y):
        T = np.size(y)-self.n
        Phi = np.zeros((T,self.n))
        for n in range(0,self.n):
            Phi[:,n] = y[n:T+n]
        return Phi
        
    def output(self):
        return self.y
    
    def log_likelihood(self, y):
        T = np.size(y)-self.n
        yhat = np.sum(self.th*self.Phi(y),1)
        return -T/2*np.log(self.sigma2) - 1/(2*self.sigma2)*np.sum(np.power(y[self.n:]-yhat,2))
       
    def posterior(self, y, m, sigma2_0): #works only for self.n = 1 so far
        Phi = self.Phi(y)
        var = 1/(1/sigma2_0+np.sum(Phi*Phi)/self.sigma2)
        return normdist((m/sigma2_0+np.sum(y[self.n:]*np.squeeze(Phi))/self.sigma2)*var,var)

N = 20
M = 50
Mc = 100
case = 2
for nt in range(0,4):
    
    if nt==0:
        T = 10
    elif nt==1:
        T = 100
    elif nt==2:
        T = 1000
    elif nt==3:
        T = 10000
    
    np.random.seed(3)
    
    if case==3:
        true_model = arm(2,np.array([1,1]),1)
        true_model.set_th(np.array([-0.3, 0.5]))
    else:
        true_model = arm(1,1,1)
        true_model.set_th(0.7)
    
    itmc = np.zeros(Mc)
    for mc in range(0,Mc):
        
        if case==2:
            true_model.simulate_with_saturation(T)
        else:
            true_model.simulate(T)
        y = true_model.output()
        
		new_model = arm(1,1,1)
            
        posterior = new_model.posterior(y,0,1)
        
        samples = posterior.sample(N)
        py = np.zeros(N)
        pyt = np.zeros([N,M])
        pytm = np.zeros(N)
        
        for i in range(0,N):
            new_model.set_th(samples[i])
            new_model.simulate(T)
            yt = new_model.output()
            py[i] = new_model.log_likelihood(y)
            for j in range(0,M):
                new_model.simulate(T)
                yt = new_model.output()
                pyt[i,j] = new_model.log_likelihood(yt)
            pyts = np.mean(py[i]>pyt[i,])
            pytm[i] = 2*np.min([pyts,1-pyts])
        itmc[mc] = np.mean(pytm)
        print(mc)
    np.savetxt("itmc_"+str(case)+"_"+str(T)+".csv", itmc, delimiter=",")