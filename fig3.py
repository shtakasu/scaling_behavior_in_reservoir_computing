import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from utils.util import *
from scipy import special
from scipy.stats import pearsonr

n_data = 100
device = "cpu"

N = 1000
Tobs = 50000
max_leadout = 100
trial = 10

if device=="cpu":
    act_func = np.tanh
elif device=="mps":
    act_func = torch.tanh

min_g = 0.2; max_g = 2.
min_sigma_n = 0. ; max_sigma_n = 3.
min_sigma_s = 0.1; max_sigma_s = 3.

Lhalf_list = np.zeros(n_data)
rho_list = np.zeros(n_data)
g_list = np.zeros(n_data)
sigma_s_list = np.zeros(n_data)
sigma_n_list = np.zeros(n_data)

ratio_list = []

for i in tqdm(range(n_data)):
    g = min_g + (max_g-min_g)*np.random.rand()
    sigma_n = min_sigma_n + (max_sigma_n-min_sigma_n)*np.random.rand()    
    sigma_s = min_sigma_s + (max_sigma_s-min_sigma_s)*np.random.rand()    
            
    g_list[i] = g; sigma_s_list[i] = sigma_s; sigma_n_list[i] = sigma_n
    
    mc_mean, _ = mc_calc(g,N,act_func,max_leadout,sigma_s, sigma_noise=sigma_n, trial=trial, Tobs= Tobs, Tinit=1000, max_delay=500, device="cpu")
    ratio = mc_mean/(np.arange(1,max_leadout+1)*mc_mean[0])
    ratio_list.append(ratio)
    for j in range(max_leadout):
        if ratio[j]<=0.5:
            Lhalf = j+1
            break
        else:
            Lhalf = max_leadout
        
    Lhalf_list[i] = Lhalf
    rho_list[i] = cor_calc(g,N,act_func, sigma_s, sigma_noise=sigma_n, trial=trial, Tobs= Tobs, Tinit=1000, device="cpu")
    
    print("******************************")
    print(f"{i+1}/{n_data} was done")
    print("******************************")
    

"""
Judge whether numerically obtained Lhalf is within the theoretical scaling regime
"""
judge_list = np.zeros(len(rho_list))

cnt=0
for g, sigma_s, sigma_n, Lhalf in zip(g_list, sigma_s_list, sigma_n_list, Lhalf_list):
    sigma_s_tilde = sigma_s*(N**0.25)
    K = 0.1
    for _ in range(1000):
        K = sigma_n**2 + sigma_s**2 + g**2 * (-1 + (4/np.pi)*np.arctan(np.sqrt(1+np.pi*K))) 
    
    alpha = np.arange(0.0, 1.01, 0.01)
    A = alpha*(sigma_s_tilde**2)/K
    B = g**2 / (1+0.5*np.pi*K)
    
    mc_theory = np.zeros(len(alpha))
    for m in range(1,100):
        mc_theory += (-1)**(m-1) * (A**m) / (1-B**m)
        
    denominator = alpha[1:]*(sigma_s_tilde**2/K)/(1-B)
    r = mc_theory[1:]/(denominator+ 1e-7)
    diverged_alpha = alpha[np.sum(np.where(r[1:]-r[:-1]<0, 1, 0))+1]
    diverged_L = diverged_alpha*np.sqrt(N)
    
    # if Lhalf < diverged_L, then the plot is assumed to be within theory (denoted by "1")
    judge_list[cnt] = 1*(Lhalf < diverged_L) 
    cnt += 1


# plot
fig = plt.figure(figsize=(7,5))
plt.scatter(rho_list[judge_list==1], Lhalf_list[judge_list==1], color="k") # a plot is black when it is within theory
plt.scatter(rho_list[judge_list==0], Lhalf_list[judge_list==0], color="r") # a plot is red when it is beyond theory
plt.xlabel("pair-wise average of "+r"$| \rho_{ij} |$",fontsize=17)
plt.ylabel(r"$L_{\rm{half}}$",fontsize=20)
plt.tick_params(labelsize=13) 
plt.ylim(0,95)
plt.yticks([0,10,20,30,40,50,60,70,80,90])
plt.savefig("fig3.png")