import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from utils.util import *
from scipy import special
import torch

g_list = [0.9, 0.9, 1.3, 1.3]
sigma_list = [0.1, 1.0, 0.1, 1.0]
sigma_noise_list = [0.1, 0.1, 0.1, 0.1]

max_leadout = 50
N=1000
Tobs = 10000
trial = 10
act_func = torch.tanh

mc_mean_list = []
mc_std_list = []
for g, sigma, sigma_noise in zip(g_list, sigma_list,sigma_noise_list):
    mc_mean, mc_std = mc_calc(g=g,N=N,act_func=act_func, max_leadout=max_leadout,Tobs=Tobs,trial=trial,sigma=sigma,max_delay=1000, device="mps", sigma_noise=sigma_noise)
    mc_mean_list.append(mc_mean)
    mc_std_list.append(mc_std)
    
fig = plt.figure(figsize=(7,5))
markers =["o", "x", "D", "s"]
i=0
for g, sigma, sigma_noise, mc_mean, mc_std in zip(g_list, sigma_list, sigma_noise_list, mc_mean_list, mc_std_list):
    plt.plot(np.arange(1,max_leadout+1), mc_mean, markersize=4, marker=markers[i], \
                label=r"$g=$"+f"{g}, " + r"$\sigma_s^2=$"+f"{sigma}"+r"$^2$")
    plt.fill_between(np.arange(1,max_leadout+1),mc_mean-mc_std, mc_mean+mc_std, alpha=0.5)
    i += 1
plt.plot(np.arange(0,len(mc_mean)+1),np.arange(0,len(mc_mean)+1),linestyle="--",color="k", label="upper bound")
plt.xlabel(r"$L$",fontsize=20)
plt.ylabel(r"$MC$",fontsize=20)
plt.tick_params(labelsize=13)
plt.xlim(0,50)
plt.ylim(0,10)
plt.legend(fontsize=14)
plt.show()