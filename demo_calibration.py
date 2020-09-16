#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Jize Zhang
"""


import numpy as np
from util_calibration import ts_calibrate, ets_calibrate, mir_calibrate, irova_calibrate
from util_evaluation import ece_eval_binary

dataset = 'cifar100'
n_class = 100
data = np.load('wideresnet-28x10cifar100epochs500.npz')

# logits: n_data * n_class matrix
logit_total = data['arr_0']

# labels: n_data * 1 matrix
label_total = data['arr_1']
# transfer labels to one-hot. If it is already one-hot, this must be skipped
label_total = np.eye(n_class)[label_total]


# Separation into calibration and evaluation set
indices = np.random.permutation(label_total.shape[0])

n_train = 10000
train_idx, test_idx = indices[:n_train], indices[10000:]


logit = logit_total[train_idx,:]
logit_eval = logit_total[test_idx,:]

label = label_total[train_idx,:]
label_eval = label_total[test_idx,:]


### For all calibration methods, the output p_eval would be the predicted probabilities

##### Method 0: Uncalibrated
print ("original uncalibrated model")
p_eval = np.exp(logit_eval)/np.sum(np.exp(logit_eval),1)[:,None]   
ece, nll, mse, accu = ece_eval_binary(p_eval,label_eval)


##### Method 1: Temperature Scaling with MSE
print ("TS with MSE Loss")
p_eval = ts_calibrate(logit,label,logit_eval, 'mse')
ece, nll, mse, accu = ece_eval_binary(p_eval,label_eval)


##### Method 2: Ensemble Temperature Scaling with MSE
print ("ETS with MSE Loss")
p_eval = ets_calibrate(logit, label, logit_eval, n_class, 'mse')
ece, nll, mse, accu = ece_eval_binary(p_eval,label_eval)


##### Method 3: Isotonic Regression (Multi-class)
print("Multi-Class Isotonic Regression")
p_eval = mir_calibrate(logit,label,logit_eval)
ece, nll, mse, accu = ece_eval_binary(p_eval,label_eval)



##### Method 4: Isotonic Regression (IROvA)
print("Isotonic Regression, OvA")
p_eval = irova_calibrate(logit,label,logit_eval)
ece, nll, mse, accu = ece_eval_binary(p_eval,label_eval)


##### Method 5: Isotonic Regression (Multi-class) with Temperature Scaling 
print("Composition: TS + Multi-Class IR")

# First layer: TS
p = ts_calibrate(logit,label,logit, 'mse')
p_eval = ts_calibrate(logit,label,logit_eval, 'mse')
# Second layer: IROvA
p_eval = mir_calibrate(np.log(p),label,np.log(p_eval))
ece, nll, mse, accu = ece_eval_binary(p_eval,label_eval)



##### Method 6: Isotonic Regression (OvA) with Temperature Scaling 
print("Composition: TS + IROvA")

# First layer: TS
p = ts_calibrate(logit,label,logit, 'mse')
p_eval = ts_calibrate(logit,label,logit_eval, 'mse')
# Second layer: IROvA
p_eval = irova_calibrate(np.log(p),label,np.log(p_eval))
ece, nll, mse, accu = ece_eval_binary(p_eval,label_eval)










