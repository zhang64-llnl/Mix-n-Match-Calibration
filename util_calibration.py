#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Jize Zhang
"""

import numpy as np
from scipy import optimize
from sklearn.isotonic import IsotonicRegression

"""
auxiliary functions for optimizing the temperature (scaling approaches) and weights of ensembles
*args include logits and labels from the calibration dataset:
"""

def mse_t(t, *args):
## find optimal temperature with MSE loss function

    logit, label = args
    logit = logit/t
    n = np.sum(np.exp(logit),1)  
    p = np.exp(logit)/n[:,None]
    mse = np.mean((p-label)**2)
    return mse


def ll_t(t, *args):
## find optimal temperature with Cross-Entropy loss function

    logit, label = args
    logit = logit/t
    n = np.sum(np.exp(logit),1)  
    p = np.clip(np.exp(logit)/n[:,None],1e-20,1-1e-20)
    N = p.shape[0]
    ce = -np.sum(label*np.log(p))/N
    return ce



def mse_w(w, *args):
## find optimal weight coefficients with MSE loss function

    p0, p1, p2, label = args
    p = w[0]*p0+w[1]*p1+w[2]*p2
    p = p/np.sum(p,1)[:,None]
    mse = np.mean((p-label)**2)   
    return mse


def ll_w(w, *args):
## find optimal weight coefficients with Cros-Entropy loss function

    p0, p1, p2, label = args
    p = (w[0]*p0+w[1]*p1+w[2]*p2)
    N = p.shape[0]
    ce = -np.sum(label*np.log(p))/N
    return ce


##### Ftting Temperature Scaling
def temperature_scaling(logit,label,loss):
    
    bnds = ((0.05, 5.0),)
    if loss == 'ce':
       t = optimize.minimize(ll_t, 1.0 , args = (logit,label), method='L-BFGS-B', bounds=bnds, tol=1e-12)
    if loss == 'mse':
        t = optimize.minimize(mse_t, 1.0 , args = (logit,label), method='L-BFGS-B', bounds=bnds, tol=1e-12)
    t = t.x
    return t



##### Ftting Enseble Temperature Scaling
def ensemble_scaling(logit,label,loss,t,n_class):

    p1 = np.exp(logit)/np.sum(np.exp(logit),1)[:,None]
    logit = logit/t
    p0 = np.exp(logit)/np.sum(np.exp(logit),1)[:,None]
    p2 = np.ones_like(p0)/n_class
    

    bnds_w = ((0.0, 1.0),(0.0, 1.0),(0.0, 1.0),)
    def my_constraint_fun(x): return np.sum(x)-1
    constraints = { "type":"eq", "fun":my_constraint_fun,}
    if loss == 'ce':
        w = optimize.minimize(ll_w, (1.0, 0.0, 0.0) , args = (p0,p1,p2,label), method='SLSQP', constraints = constraints, bounds=bnds_w, tol=1e-12, options={'disp': True})
    if loss == 'mse':
        w = optimize.minimize(mse_w, (1.0, 0.0, 0.0) , args = (p0,p1,p2,label), method='SLSQP', constraints = constraints, bounds=bnds_w, tol=1e-12, options={'disp': True})
    w = w.x
    return w





"""
Calibration: 
Input: uncalibrated logits, temperature (and weight)
Output: calibrated prediction probabilities
"""

##### Calibration: Temperature Scaling with MSE
def ts_calibrate(logit,label,logit_eval,loss):
    t = temperature_scaling(logit,label,loss)
    print("temperature = " +str(t))
    logit_eval = logit_eval/t
    p = np.exp(logit_eval)/np.sum(np.exp(logit_eval),1)[:,None]   
    return p


##### Calibration: Ensemble Temperature Scaling
def ets_calibrate(logit,label,logit_eval,n_class,loss):
    t = temperature_scaling(logit,label,loss='mse') # loss can change to 'ce'
    print("temperature = " +str(t))
    w = ensemble_scaling(logit,label,'mse',t,n_class)
    print("weight = " +str(w))



    p1 = np.exp(logit_eval)/np.sum(np.exp(logit_eval),1)[:,None]
    logit_eval = logit_eval/t
    p0 = np.exp(logit_eval)/np.sum(np.exp(logit_eval),1)[:,None]
    p2 = np.ones_like(p0)/n_class
    p = w[0]*p0 + w[1]*p1 +w[2]*p2
    return p



##### Calibration: Isotonic Regression (Multi-class)
def mir_calibrate(logit,label,logit_eval):
    p = np.exp(logit)/np.sum(np.exp(logit),1)[:,None] 
    p_eval = np.exp(logit_eval)/np.sum(np.exp(logit_eval),1)[:,None]
    ir = IsotonicRegression(out_of_bounds='clip')
    y_ = ir.fit_transform(p.flatten(), (label.flatten()))
    yt_ = ir.predict(p_eval.flatten())
    
    p = yt_.reshape(logit_eval.shape)+1e-9*p_eval
    return p

def irova_calibrate(logit,label,logit_eval):
    p = np.exp(logit)/np.sum(np.exp(logit),1)[:,None] 
    p_eval = np.exp(logit_eval)/np.sum(np.exp(logit_eval),1)[:,None]
    

    for ii in range(p_eval.shape[1]):
        ir = IsotonicRegression(out_of_bounds='clip')
        y_ = ir.fit_transform(p[:,ii], label[:,ii])
        p_eval[:,ii] = ir.predict(p_eval[:,ii])+1e-9*p_eval[:,ii]
    return p_eval
    return p_eval