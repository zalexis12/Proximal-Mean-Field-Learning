#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 12 21:05:24 2023

@author: ateter1
"""

def Drift():
    w = theta_prev
    w = w.requires_grad_()
    mult = torch.matmul(x, w)    
    Phi_all = torch.softmax(mult, axis=0)
  #  print(Phi_all.shape)
    #print(y_full.shape)
    Phi_correct = torch.matmul(torch.transpose(torch.transpose(Phi_all, 1,2), 0,2),y_full)
    Phi_correct = Phi_correct.reshape(nMiniData, nSample)
    Phi_correct = Phi_correct*scale
    U = 1/nMiniData*torch.transpose(Phi_correct, 0, 1).mm(Phi_correct)
    V_prev = -2/nMiniData*torch.transpose(Phi_correct, 0,1).mm(scale*torch.ones((nMiniData,1), device=device, dtype=dtype))
    drift = U.mm(rho_prev) + V_prev
    drift.backward(one)
    driftk = - w.grad.data
    Phi_correct = Phi_correct.detach()
    U = U.detach()
    V_prev = V_prev.detach()
    w=w.detach()
    return driftk, Phi_correct, U, V_prev

###########################################
#Fixed Point Iteration:
###########################################
def FixedPointIteration():
    tol=1e-3  #tolerance

    #     squared distance matrix
    #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!1
    #ADJUST TO BE SUM TO GET 2D MATRIX INSTEAD OF #D
    C = torch.zeros((nSample,nSample), device = device, dtype=dtype)
    for i in range(10):
        C = C + torch.cdist(torch.transpose(theta_current[i,:,:],0,1), torch.transpose(theta_prev[i,:,:],0,1), p=2)**2
   # C=C/10
    gamma = torch.exp(-C/(2*epsilon));

    xi=torch.exp(-beta*(V_prev)-beta*torch.matmul(U_prev,rho_prev)-one)

    lambda_1=torch.rand((nSample,1), device=device, dtype=dtype)
    z0=torch.exp(lambda_1*h/epsilon)

    z =torch.hstack((z0,zero_torch))

    yy=torch.hstack((rho_prev/(torch.matmul(gamma,z0)),zero_torch))

    l=0
    while l < maxiter-1:

        z[:,l+1]=torch.pow(xi/(torch.matmul(torch.transpose(gamma, 0, 1),yy[:,l].reshape(nSample,1))),1/(1+(beta*epsilon/h)))[:,0]

        yy[:,l+1]=(rho_prev/(torch.matmul(gamma,z[:, l+1].reshape(nSample,1))))[:,0]
        norm_y = (yy[:, l+1]-yy[:, l]).pow(2).sum()
        norm_z = (z[:, l+1]-z[:, l]).pow(2).sum()

        if (norm_y < tol):
            if (norm_z<tol):
                break

        l+=1

    rho_next=z[:,l].reshape(nSample,1)*(torch.matmul(torch.transpose(gamma,0,1),yy[:,l].reshape(nSample,1)))

    return rho_next

###########################################
#Euler Maruyama:
###########################################
def EulerMaruyama_MeanField():
    gdw = math.sqrt(h*2/beta)*torch.randn((10, nx, nSample), device=device, dtype=dtype)/100
    theta_current = theta_prev + h*driftk + gdw

    return theta_current

import numpy as np
import torch
from torch.autograd import Variable
import time
import math
import scipy.io
from pycuda.compiler import SourceModule
import pycuda.driver as cuda
import pycuda.autoinit
from torch import linalg as LA


###########################################
#Setting up GPU:
###########################################

torch.set_printoptions(precision=8) 
dtype = torch.float32
device = torch.device("cuda:0")

data_all = np.genfromtxt('semeion.data')
data_all_test = data_all[1000:,:]
data_all = data_all[0:1000,:]
data = data_all[:, 0:256]
data_test = data_all_test[:, 0:256]
data=data-0.5
data=data*2
data_test=data_test-0.5
data_test=data_test*2
labels_full = data_all[:,256:266]
labels_full_test = data_all_test[:,256:266]

labels = np.argmax(labels_full, axis=1)
labels_test = np.argmax(labels_full_test, axis=1)

data_torch = torch.tensor(data, device=device, dtype=dtype)
labels_torch = torch.tensor(labels, device=device, dtype=dtype)
labels_full = torch.tensor(labels_full, device=device, dtype=dtype)

data_torch_test = torch.tensor(data_test, device=device, dtype=dtype)
labels_torch_test = torch.tensor(labels_test, device=device, dtype=dtype)
labels_full_test = torch.tensor(labels_full_test, device=device, dtype=dtype)


###########################################
#Setting constant variables and 
#initializing weights/rho,
#as well as creating storage vectors
#for various Risk/likelihood values
#to track:
###########################################

nSample=100 # number of samples
beta=.5 # inverse temperature
epsilon=10  #entropic regularizing coefficient
h=.001 # time step
numSteps=int(1000000) #1e6 # number of steps k, in discretization t=kh
w_min, w_max=-1,1;
nx = 256;
nData = 1000
nData_test = 593
nMiniData = 1000 #Number of data points in each minibatch
count_run = 0

scale=1
labels_full = labels_full.reshape((nData, 10, 1))
labels_full_test = labels_full_test.reshape((nData_test, 10, 1))

#Risk=torch.zeros((numSteps, 1), device=device, dtype=dtype)
#Risk_weight=torch.zeros((numSteps, 1), device=device, dtype=dtype)

Risk_test=torch.zeros((numSteps, 1), device=device, dtype=dtype)
Risk_weight_test=torch.zeros((numSteps, 1), device=device, dtype=dtype)

theta_prev = w_min+(w_max-w_min)*torch.rand((10, nx, nSample), device=device, dtype=dtype)
rho_prev = torch.ones((nSample,1), device=device, dtype=dtype)*1/100#/scale

#theta_prev_save = theta_prev.to("cpu")
#rho_prev_save = rho_prev.to("cpu")

#theta_prev_save = theta_prev_save.detach().numpy()
#rho_prev_save = rho_prev_save.detach().numpy()

#for i in range(10):
   # np.savetxt('theta_save_semeion' + str(i) + '.dat', theta_prev_save[i,:,:])

#np.savetxt('rho10D_save' + '.dat', rho_prev_save)

#theta_prev = torch.zeros((10, nx, nSample), device=device, dtype=dtype)
#for i in range(10):
  #  theta_hold = np.genfromtxt('theta_save_semeion' + str(i) + '.dat')
 #   theta_prev[i,:,:] = torch.tensor(theta_hold, device=device, dtype=dtype)
#rho_prev = np.genfromtxt('rho10D_save.dat')
#rho_prev = rho_prev.reshape([nSample,1])

rho_prev = torch.tensor(rho_prev, device=device, dtype=dtype)
one = torch.ones((nSample,1), device=device, dtype=dtype) 
one_test = torch.ones((nData_test,1), device=device, dtype=dtype)  
maxiter=300 # max number of iterations for k fixed
zero_torch = torch.zeros((nSample,maxiter - 1), device=device, dtype=dtype)

x = data_torch
y = labels_torch
y_full = labels_full

###########################################
#Use a for loop to run through the
#set number of iterations:
###########################################
t1 = time.time()
for k in range(numSteps):

	###########################################
	#Update rho and weights:
	###########################################
    
    driftk, Phi, U_prev, V_prev = Drift()
    theta_prev = theta_prev.detach()
    theta_current=EulerMaruyama_MeanField()
    
    rho_next = FixedPointIteration()
      
	###########################################
	#Update previous values:
	###########################################
	
    theta_prev=theta_current

    rho_prev=rho_next

	###########################################
	#Calculate and store risk values:
	###########################################
    
    mult_test = torch.matmul(data_torch_test, theta_prev)    
    Phi_all_test = torch.softmax(mult_test, axis=0)

    Phi_correct_test = torch.matmul(torch.transpose(torch.transpose(Phi_all_test, 1,2), 0,2),labels_full_test)
    Phi_correct_test = Phi_correct_test.reshape(nData_test, nSample)
    Risk_test[k] = torch.norm(one_test-torch.matmul(Phi_correct_test, one)/nSample, p=2)/nData_test
    Risk_weight_test[k] = torch.norm(one_test-torch.matmul(Phi_correct_test, rho_prev), p=2)/nData_test
    
    to_store = k%1000
    run=0
    if (to_store == 0):

            Phi_risk = torch.matmul(data_torch, theta_prev)
            Phi_correct = (torch.argmax(Phi_risk, dim=0)).float()
            for i in range(nSample):
                bool_correct = torch.eq(Phi_correct[:,i], labels_torch)
                Phi_correct[:,i] = bool_correct.float()

            #Risk_value = torch.sum(torch.sum(Phi_correct, axis=0))/nSample

            print("After" + str(k) + " full runs, the test (weighted) is" + str(Risk_weight_test[k]) + "unweighted" + str(Risk_test[k]))
            t2=time.time()  
            print("Total time passed is:" +  str(t2 -t1))

            
###########################################
#Save the risks, current weights and rho
###########################################
theta_prev = theta_prev.to("cpu")
rho_prev = rho_prev.to("cpu")

theta_prev = theta_prev.detach().numpy()
rho_prev = rho_prev.detach().numpy()

for i in range(10):
    np.savetxt('theta10D_5semeion' + str(i) + '.dat', theta_prev[i,:,:])

np.savetxt('rho10D_5semeion' + '.dat', rho_prev)

Risk_test = Risk_test.to("cpu")
Risk_weight_test =  Risk_weight_test.to("cpu")

Risk_test = Risk_test.detach().numpy()
Risk_weight_test = Risk_weight_test.detach().numpy()

np.savetxt('Risk_semeion4.dat', Risk_test)
np.savetxt('Risk_weight_hard_semeion4.dat', Risk_weight_test)