#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 12 17:35:55 2023

@author: ateter1
"""

def Drift():
    w= theta_prev[:,2:]
    a= theta_prev[:,0].reshape(nSample,1)
    b= theta_prev[:,1].reshape(nSample,1)
    a = a.requires_grad_()
    b = b.requires_grad_()
    w = w.requires_grad_()    

    Phi_torch = a*(torch.tanh(w.mm(torch.transpose(x_torch,0,1))+b))
    U_prev=(1/nData)*Phi_torch.mm(torch.transpose(Phi_torch, 0, 1))
    W_prev=U_prev.mm(rho_prev)
    V_prev=(-2/nData)*Phi_torch.mm(y_torch)
    drift=V_prev+W_prev
    drift.backward(one)

    driftk = torch.hstack((-a.grad.data,-b.grad.data,-w.grad.data))
    Phi_torch = Phi_torch.detach()
    U_prev = U_prev.detach()
    V_prev = V_prev.detach()
    return driftk, Phi_torch, U_prev, V_prev

#####################################################################################################################
def FixedPointIteration():
    global bool_cpu
    global bool_cpu_2
    #global bool_storage
    #global bool_storage_2
    tol=1e-3  #tolerance

    #     squared distance matrix

    C = torch.cdist(theta_current, theta_prev, p=2)**2

    #     elementwise exponential of a matrix
    gamma = torch.exp(-C/(2*epsilon));
    #     elementwise exponential of a vector

    xi=torch.exp(-beta*(V_prev)-beta*torch.matmul(U_prev,rho_prev)-one)

    lambda_1=torch.rand((nSample,1), device=device, dtype=dtype)
    z0=torch.exp(lambda_1*h/epsilon)
    z =torch.hstack((z0,zero_torch))

    yy=torch.hstack((rho_prev/(torch.matmul(gamma,z0)),zero_torch))

    l=0
    while l < maxiter-1:
        # if l>1:

        z[:,l+1]=torch.pow(xi/(torch.matmul(torch.transpose(gamma, 0, 1),yy[:,l].reshape(nSample,1))),1/(1+(beta*epsilon/h)))[:,0]

        yy[:,l+1]=(rho_prev/(torch.matmul(gamma,z[:, l+1].reshape(nSample,1))))[:,0]
        yy_storage = yy[:, l+1]-yy[:, l]
        zz_storage = z[:, l+1]-z[:, l]

        yy_storage = yy_storage.pow(2)
        zz_storage = zz_storage.pow(2)
        norm_y = torch.sqrt(yy_storage.sum())
        norm_z = torch.sqrt(zz_storage.sum())

        bool_cpu = (norm_y < tol).to("cpu").numpy()

        bool_cpu_2 = ((norm_z<tol).to("cpu")).numpy()

        if (bool_cpu_2):
            if (bool_cpu):
                break

        l+=1
        
    rho_next=z[:,l].reshape(nSample,1)*(torch.matmul(torch.transpose(gamma,0,1),yy[:,l].reshape(nSample,1)))
    return rho_next

#####################################################################################################################
def EulerMaruyama_MeanField():
    gdw = math.sqrt(h*2/beta)*torch.randn((nSample,n_p), device=device, dtype=dtype)
    theta_current = theta_prev + h*driftk + gdw
    return theta_current
#####################################################################################################################

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

torch.set_printoptions(precision=8)
raw_data = scipy.io.loadmat('diabetes.mat')

##Training data (e.g, wbcd dataset)
# ===========================================
#  Wisconsin breast cancer data
#  source: https://www.kaggle.com/uciml/breast-cancer-wisconsin-data
data = raw_data['diabetes']
nData = len(data)
#print(data.type)
np.random.shuffle(data)

 #number of datapoints available in the training data

spliting_persentage=50

train_data = data[:math.ceil((1-spliting_persentage/100)*nData)]
test_data = data[math.ceil((1-spliting_persentage/100)*nData):]
nData_test=len(test_data)

data=train_data
nData=len(train_data)
print(nData)
print(nData_test)
#  binary label vector
labels = data[:,1].reshape(nData,1)

# rescale labels from {0,1} to {-1,1}
ymin, ymax= -1, 1
y = ymin*np.ones((len(data),1)) + (ymax - ymin)*labels
#  features
x=data[:,2:]*8
for i in range(8):
    x[:,i]=x[:,i]/np.max(x[:,i])
#x=x*3000    
nx = len(x[0])

dtype = torch.float32
device = torch.device("cuda:0")
    
# the feature and the labels
x_torch = torch.tensor(x, device=device, dtype=dtype)
y_torch = torch.tensor(y, device=device, dtype=dtype)

labels_test =test_data[:,1].reshape(nData_test,1)

# rescale labels from {0,1} to {-1,1}
y_test = ymin*np.ones((len(test_data),1)) + (ymax - ymin)*labels_test

labels = torch.tensor(labels, device=device, dtype=dtype)
labels_test = torch.tensor(test_data[:,1].reshape(nData_test,1), device=device, dtype=dtype)

#  features
x_test=test_data[:,2:]*8
for i in range(8):
    x_test[:,i]=x_test[:,i]/np.max(x_test[:,i])

#x_test=x_test*3000
nx_test = len(x_test[0])
print(np.max(x))
print(np.max(x_test))
# the feature and the labels
x_test_torch = torch.tensor(x_test, device=device, dtype=dtype)
y_test_torch = torch.tensor(y_test, device=device, dtype=dtype)

n_p=nx+2


nSample=1000 # number of samples
beta=.65 # inverse temperature
epsilon=1  #entropic regularizing coefficient
h=1e-4 # time step
numSteps=int(5e5) #1e6 # number of steps k, in discretization t=kh

a_min, a_max=0.9, 1.1; # min-max scaling
b_min, b_max=-0.1, 0.1; # min-max bias
w_min, w_max=-2, 2;
numRandomRun=1 # 50
lambda_1=torch.rand((nSample,1), device=device, dtype=dtype)
Risk_test=torch.zeros((numSteps,1), device=device, dtype=dtype)
Risk_weight_hard_test=torch.zeros((numSteps,1), device=device, dtype=dtype)

Risk=torch.zeros((numSteps, 1), device=device, dtype=dtype)
Risk_weight_hard=torch.zeros((numSteps, 1), device=device, dtype=dtype)

F0=np.sum(y**2)/nData

one=torch.ones((nSample,1), device=device, dtype=dtype)
a=a_min+(a_max-a_min)*torch.rand((nSample,1), device=device, dtype=dtype)

bool_cpu = (LA.norm(one) > LA.norm(a)).to("cpu").numpy()
bool_cpu_2 = (LA.norm(one) > LA.norm(a)).to("cpu").numpy()

bool_cpu = (LA.norm(one) > LA.norm(a)).to("cpu").numpy()
bool_cpu_2 = (LA.norm(one) > LA.norm(a)).to("cpu").numpy()


for r in range(numRandomRun):
    a=a_min+(a_max-a_min)*torch.rand((nSample,1), device=device, dtype=dtype)
    b=b_min+(b_max-b_min)*torch.rand((nSample,1), device=device, dtype=dtype)
    w=w_min+(w_max-w_min)*torch.rand((nSample,nx), device=device, dtype=dtype)

    yy_storage = torch.ones((nSample), device=device, dtype=dtype)
    zz_storage = torch.ones((nSample), device=device, dtype=dtype)
    maxiter=300 # max number of iterations for k fixed
    zero_torch = torch.zeros((nSample,maxiter - 1), device=device, dtype=dtype)
    #      concatenate
    theta_prev=torch.hstack((a,b,w))

    rho_prev=torch.ones((nSample,1), device=device, dtype=dtype)/1000#*(1/((a_max-a_min)*(b_max-b_min)*((w_max-w_min)**nx)))

    tol_risk=1e-3 # numbers coloser than this number to zero and one 
                  # in the f_hat_weight_soft and f_hat_soft will be considered as zero and one, respectively.
    f_hat_weight_hard_test=torch.zeros((nData_test,1), device=device, dtype=dtype)
    f_hat_soft_test=torch.rand((nData_test,1), device=device, dtype=dtype)

    f_hat_weight_hard=torch.zeros((nData,1), device=device, dtype=dtype)
    f_hat_soft=torch.rand((nData,1), device=device, dtype=dtype)
    
    mod = SourceModule("""
                    __global__ void f_hat_soft_test_upd(float *f_hat_soft, int N)
                    {
                            int idx = threadIdx.x + blockDim.x * blockIdx.x;
                            float tol_risk = 1.0e-3;
                            if (idx < N) {
                                    if (f_hat_soft[idx] > 1 - tol_risk){
                                            f_hat_soft[idx] = 1;
                                    }
                                    if (f_hat_soft[idx] < tol_risk){
                                            f_hat_soft[idx] = 0;
                                    }
                            }
                   }
                   """)
    func_soft_test = mod.get_function("f_hat_soft_test_upd")

    mod = SourceModule("""
                    __global__ void f_hat_weight_test_upd(float *f_hat_weight, int N)
                    {
                            int idx = threadIdx.x + blockDim.x * blockIdx.x;
                            if (idx < N) {
                                    if (f_hat_weight[idx] > 0){
                                            f_hat_weight[idx] = 1;
                                    }
                                    if (f_hat_weight[idx] < 0){
                                            f_hat_weight[idx] = 0;
                                    }
                            }
                   }
                   """)
    func_weight_test = mod.get_function("f_hat_weight_test_upd")
    time1 = time.time()
    driftk, Phi, U_prev, V_prev=Drift()

        #        update of the parameter samples

    
    theta_current=EulerMaruyama_MeanField()

        #       proximal update of the joint PDF
    rho_next=FixedPointIteration()

    
    theta_prev=theta_current
    rho_prev=rho_next
    for k in range(numSteps):

        driftk, Phi, U_prev, V_prev=Drift()

        #        update of the parameter samples


        theta_current=EulerMaruyama_MeanField()

        #       proximal update of the joint PDF
        rho_next=FixedPointIteration()

        theta_prev=theta_current
        rho_prev=rho_next
        Phi_test=(torch.tanh(torch.matmul(theta_prev[:,2:],torch.transpose(x_test_torch, 0, 1))+theta_prev[:,1].reshape(nSample,1)))*(theta_prev[:,0].reshape(nSample,1))
        f_hat_test=Phi_test.transpose(0,1).sum(axis=1).reshape(nData_test,1)/nSample
        f_hat_soft_test=torch.exp(f_hat_test)/(1+torch.exp(f_hat_test))


        ## computing the risk function with weighted f_hat for test data
        f_hat_weight_test=torch.matmul(Phi_test.transpose(0,1),rho_prev)
        func_soft_test(f_hat_soft_test, np.int64(nData_test), block=(77,1,1), grid=(1,1,1))
        func_weight_test(f_hat_weight_test.detach(), np.int32(nData_test), block=(77,1,1), grid=(1,1,1))
        Risk_test[k]=(0.5/nData_test)*torch.linalg.norm(labels_test-f_hat_soft_test)
        Risk_weight_hard_test[k]=(0.5/nData_test)*torch.norm(labels_test-f_hat_weight_test,2)
        f_hat=Phi.transpose(0,1).sum(axis=1).reshape(nData,1)/nSample
        f_hat_soft=(torch.exp(f_hat))/(1+torch.exp(f_hat))
        ## computing the risk function with weighted f_hat for test data
        f_hat_weight=torch.matmul(Phi.transpose(0,1),rho_prev)  

        func_soft_test(f_hat_soft.detach(), np.int32(nData), block=(691,1,1), grid=(1,1,1))
        func_weight_test(f_hat_weight.detach(), np.int32(nData), block=(691,1,1), grid=(1,1,1))

        Risk[k]=(0.5/nData)*torch.norm(labels-f_hat_soft,2)
        Risk_weight_hard[k]=(0.5/nData)*torch.norm(labels-f_hat_weight,2)

        if (k%1e3 == 0):
            print("Now, running iterations between k=",k+1,"and k=", k+int(1e3))
            print("And, the Risk value at iteration #",k, "was:",Risk_test[k-1])
            print("Computation time: "+str(time.time() - time1))
            print(f_hat_weight_test.shape)
            for i in range(nData):
                if f_hat_weight[i] > 0:
                    f_hat_weight[i]=1
                if f_hat_weight[i] < 0:
                    f_hat_weight[i]=0
                
            for i in range(nData_test):
                if f_hat_weight_test[i] > 0:
                    f_hat_weight_test[i]=1
                if f_hat_weight_test[i] < 0:
                    f_hat_weight_test[i]=0
        
            for i in range(nData):
                if f_hat[i] > 0:
                    f_hat[i]=1
                if f_hat[i] < 0:
                    f_hat[i]=0
                
            for i in range(nData_test):
                if f_hat_test[i] > 0:
                    f_hat_test[i]=1
                if f_hat_test[i] < 0:
                    f_hat_test[i]=0
            acc = sum(f_hat_weight_test==labels_test)
            print(acc/nData_test)
            acc = sum(f_hat_weight==labels)
            print(acc/nData)
            acc = sum(f_hat_test==labels_test)
            print(acc/nData_test)
            acc = sum(f_hat==labels)
            print(acc/nData)

theta_prev = theta_prev.to("cpu")
rho_prev = rho_prev.to("cpu")
theta_prev = theta_prev.detach().numpy()
rho_prev = rho_prev.detach().numpy()
np.savetxt('theta.dat', theta_prev)
np.savetxt('rho.dat', rho_prev)
 
comptime = time.time()
print("Computation time: "+str(comptime - time1))

Risk_test = Risk_test.to("cpu")
Risk_weight_hard_test =  Risk_weight_hard_test.to("cpu")
Risk_test = Risk_test.detach().numpy()
Risk_weight_hard_test = Risk_weight_hard_test.detach().numpy()
np.savetxt('Risk_test.dat', Risk_test)
np.savetxt('Risk_weight_hard_test.dat', Risk_weight_hard_test)

Risk = Risk.to("cpu")
Risk_weight_hard =  Risk_weight_hard.to("cpu")

Risk = Risk.detach().numpy()
Risk_weight_hard = Risk_weight_hard.detach().numpy()
np.savetxt('Risk.dat', Risk)
np.savetxt('Risk_weight_hard.dat', Risk_weight_hard)