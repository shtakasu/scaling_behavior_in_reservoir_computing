import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from utils.ipc import *
from utils.util import *

N = 1000
g = 1.
noise_std = 0.3
input_intensity = 1.
max_leadout = 50
degree_list = [1,2,3,4,5]#[1,3,5,7,9]
Tobs = int(1e5) #observation time length
Tinit = int(1e4) #washout time length
act_func =  lambda x: np.maximum(0,x) #ReLU activation function
trial = 10

ipc_list = np.zeros([trial, max_leadout, len(degree_list)])
for i in range(trial):  
    ipc = calc_ipc(N,g,input_intensity,max_leadout,degree_list,Tobs,Tinit,act_func,noise_std=noise_std)
    ipc_list[i,:,:] = ipc.copy()
    print(f"trial {i+1}/{trial} was done")
    
# top raw figure   
fig = plt.figure(figsize=(7,5))
hatch_list  = ['//', 'x', '||', '\\', "+", "..", "xx"]
bottom = np.zeros([max_leadout])
upper = np.mean(ipc_list[:,:,0], axis=0)

for i in range(1, len(degree_list)+1):
    plt.fill_between(np.arange(1,max_leadout+1), bottom, upper, label=f"D={degree_list[i-1]}", hatch=hatch_list[i-1])
    bottom += np.mean(ipc_list[:,:,i-1], axis=0)
    if i<len(degree_list):
        upper += np.mean(ipc_list[:,:,i], axis=0)
plt.plot(np.arange(1,max_leadout+1),np.arange(1,max_leadout+1), color="black",linestyle="--")
plt.xticks([10,20, 30, 40, 50])
#plt.xlabel(r"$L$",fontsize=16)
plt.ylabel(r"$IPC_{total}$",fontsize=25)
plt.tick_params(labelsize=15)
plt.xlim(0,max_leadout)
plt.ylim(0,max_leadout)
plt.savefig("fig8top.png")

# bottom raw figure  
fig = plt.figure(figsize=(7,4))
markers =["o", "x", "D", "s", "^", ">"]
for i in range(0, len(degree_list)):
    plt.plot(np.arange(1,max_leadout+1), np.mean(ipc_list[:,:,i], axis=0), label=f"D={degree_list[i]}", marker=markers[i], markersize=4)
    plt.fill_between(np.arange(1,max_leadout+1), np.mean(ipc_list[:,:,i], axis=0)-np.std(ipc_list[:,:,i], axis=0),\
                                                np.mean(ipc_list[:,:,i], axis=0)+np.std(ipc_list[:,:,i], axis=0), alpha=0.3)
plt.xticks([10,20, 30, 40, 50])
plt.xlabel(r"$L$",fontsize=25)
plt.ylabel(r"$IPC_D$",fontsize=25)
plt.tick_params(labelsize=15)
plt.xlim(0,max_leadout)
plt.ylim(0,15)
plt.axhline(y=0, color="k", linestyle="--")
#plt.legend(fontsize=18)
plt.savefig("fig8bottom.png")