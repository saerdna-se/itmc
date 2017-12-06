# -*- coding: utf-8 -*-
"""
Source code for the Ljung-Box part of the fourth synthetic example (Repeated evaluation of Case iv & v, Figure 6) in
"Is My Model Flexible Enough? Information-Theoretic Model Check",
Andreas Svensson, Dave Zachariah, Thomas B. Schön arXiv:NNNN

Code by Andreas Svensson 2017
"""
import numpy as np
from scipy.stats import norm
from scipy.stats import chi2
from scipy import signal
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
    
    def maximum_likelihood(self, y): #works only for self.n = 1 so far
        Phi = self.Phi(y)
        return np.sum(y[self.n:]*np.squeeze(Phi))/np.sum(Phi*Phi)
    
    def one_step_ahead_predict(self,y):
        yp = np.zeros(T)
        yp[0:self.n] = self.y0
        for t in range(self.n,T):
            yp[t] = np.sum(self.th*y[t-self.n:t])
        return yp
            
    def prediction_error(self,y):
        yp = self.one_step_ahead_predict(y)
        ye = yp-y
        return ye
    
    def prediction_error_corr(self,y):
        ye = self.prediction_error(y)
        p = signal.correlate(ye,ye)
        return p/np.max(p)

Mc = 100
for case in range(4,6):
    T = 100
    np.random.seed(3)
    
    if case==4:
        true_model = arm(1,1,.1)
    else:
        true_model = arm(1,1,1)
	true_model.set_th(0.7)
    
    lb = np.zeros(Mc)
    for mc in range(0,Mc):
        
        true_model.simulate(T)
        y = true_model.output()
        
        if case==5:
            new_model = arm(1,1,.1)
        else:
            new_model = arm(1,1,1)
            
        ml_th = new_model.maximum_likelihood(y)
        
        new_model.set_th(ml_th)
        
        p = new_model.prediction_error_corr(y)
        
        h = int(np.max([np.round(np.log(T)),3]))
        
        p2 = np.power(p[int(np.ceil(p.size/2)):int(np.ceil(p.size/2)+h)],2)
        Q = T*(T+2)*sum(p2/np.linspace(T-1,T-h,h))
        lb[mc] = (1-chi2.cdf(Q,df=h-1))
    np.savetxt("lb_"+str(ex)+"_"+str(T)+".csv", lb, delimiter=",")
