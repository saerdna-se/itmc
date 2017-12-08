# -*- coding: utf-8 -*-
"""
Source code for the first synthetic example (dispersion of Case i & ii, Figure 1 & 2) in
"Is My Model Flexible Enough? Information-Theoretic Model Check",
Andreas Svensson, Dave Zachariah, Thomas B. Schön arXiv:1712.02675

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
Tmax = 500

for case in range(1,3):
#case = 0 #1 = miss-specified, 0 = well-specified

    Tvec= np.arange(3,Tmax+1,1)
    
    Nt = np.size(Tvec)
    
    np.random.seed(2)
    true_model = arm(1,1,1)
    true_model.set_th(0.7)
    if case==2:
        true_model.simulate_with_saturation(Tmax)
    else:
        true_model.simulate(Tmax)
    y = true_model.output()
    itmc = np.zeros(Nt)
    dispersion_square = np.zeros(Nt)
    
    py = np.zeros(N)
    pyt = np.zeros([N,M])
    pytm = np.zeros(N)
    
    
    for nt in range(0,Nt):
        np.random.seed(1)
        
        T = int(Tmax/Nt*(nt+1))
        yT = y[0:T]
        
        new_model = arm(1,1,1)
        posterior = new_model.posterior(yT,0,1)
        
        samples = posterior.sample(N)
        
        for i in range(0,N):
            new_model.set_th(samples[i])
            new_model.simulate(T)
            yt = new_model.output()
            py[i] = new_model.log_likelihood(yT)
            for j in range(0,M):
                new_model.simulate(T)
                yt = new_model.output()
                pyt[i,j] = new_model.log_likelihood(yt)
            pyts = np.mean(py[i]>pyt[i,])
            pytm[i] = 2*np.min([pyts,1-pyts])
        itmc[nt] = np.mean(pytm)
        dispersion_square[nt] = np.mean(np.square(pytm-itmc[nt]))
        print(nt)
    plt.plot(Tvec,itmc,'r')
    plt.plot(Tvec,itmc+np.sqrt(dispersion_square),'g')
    plt.plot(Tvec,itmc-np.sqrt(dispersion_square),'g')
    res = np.transpose(np.array([Tvec,itmc,dispersion_square]))
    np.savetxt("res_"+str(case)+"b.csv", res, delimiter=' ', newline='\n', fmt='%10.5f')
    np.savetxt("y_"+str(case)+"b.csv", y, delimiter=' ', newline='\n')
