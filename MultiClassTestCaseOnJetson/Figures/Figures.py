import numpy as np
# from ttictoc import tic,toc
import matplotlib.pyplot as plt
import pylab
from numpy import linalg as LA
# import tensorflow as tf
import math
import scipy.io

params = {'backend': 'ps',
          'xtick.labelsize': 12,
          'ytick.labelsize': 12,
          'legend.handlelength': 1,
          'legend.borderaxespad': 0,
          'font.family': 'serif',
          'font.serif': ['Computer Modern Roman'],
          'ps.usedistiller': 'xpdf',
          'text.usetex': True,
          # include here any neede package for latex
          'text.latex.preamble': [r'\usepackage{amsmath}'],
          }
plt.rcParams.update(params)


# Risk_weight = np.zeros((1000000,30))

# for i in range(30):
#     Risk_weight[:,i] = np.genfromtxt('Risk_weight_' + str(i+1) + '.dat')


# min_risk_weight=np.minimum.reduce(np.transpose(Risk_weight))


# max_risk_weight=np.maximum.reduce(np.transpose(Risk_weight))


# # print(min_risk_weight.shape)


# np.savetxt("min_risk_weight.dat",min_risk_weight)


# np.savetxt("max_risk_weight.dat",max_risk_weight)





min_risk_weight=np.genfromtxt("min_risk_weight.dat")
max_risk_weight=np.genfromtxt("max_risk_weight.dat")
mean_Risk_weight=np.genfromtxt("mean_Risk_weight.dat")




min_risk=np.genfromtxt("min_risk.dat")
max_risk=np.genfromtxt("max_risk.dat")
mean_Risk=np.genfromtxt("mean_Risk.dat")




# min_risk_weight=np.power(min_risk_weight,2)
# max_risk_weight=np.power(max_risk_weight,2)
# mean_Risk_weight=np.power(mean_Risk_weight,2)




# min_risk=np.power(min_risk,2)
# max_risk=np.power(max_risk,2)
# mean_Risk=np.power(mean_Risk,2)






x = np.arange(len(min_risk))


fig2 = plt.figure()
ax = fig2.add_subplot(2, 1, 1)

line, = ax.loglog(mean_Risk, color='red', lw=1.5,linestyle='--',label="Unweighted")
line, = ax.step(0,0, color='blue', lw=1.5, label="Weighted")

plt.fill_between(x, min_risk, max_risk, color='red',alpha=.3)


ax.set_ylabel(r"Risk functional $F_{\beta}$",fontsize=15)
ax.yaxis.set_label_coords(-0.11,-.1)



ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.legend(markerscale=1.5, numpoints=1, loc='upper center', ncol=2,  prop={'size':15}, frameon=False,bbox_to_anchor=[0.5, 1.2], columnspacing=1.2, labelspacing=0.1, handletextpad=0.5, handlelength=2)
ax.grid(True, which="both", ls="-", color='0.8')
ax.set_yticks([.5,.1])
ax = fig2.add_subplot(2, 1, 2)

line, = ax.loglog(mean_Risk_weight, color='blue', lw=1.5, label="weighted")

plt.fill_between(x, min_risk_weight, max_risk_weight, color='blue',alpha=.3)


ax.yaxis.set_label_coords(-0.11,0.45)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
# ax.legend(markerscale=1.5, numpoints=1, loc='upper center', ncol=2,  prop={'size':15}, frameon=False,bbox_to_anchor=[0.5, 1.1], columnspacing=1.2, labelspacing=0.0, handletextpad=0.2, handlelength=2)
ax.grid(True, which="both", ls="-", color='0.8')
fig2.set_size_inches(8, 5 )
ax.set_xlabel(" Iterations",fontsize=15)
plt.savefig('Risk_semeion_double.png', dpi=300)


