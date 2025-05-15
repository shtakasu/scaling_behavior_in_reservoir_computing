import numpy as np
import matplotlib.pyplot as plt

d_alpha = 0.001
alpha = np.arange(0.001, 1.001, d_alpha)
sigma_s_tilde_list = np.arange(0.4, 1.1, 0.1)
g = 1.2
sigma_n = 0.5


r_list = []
r_nondiverge_list = []
threshold_list = []

fig = plt.figure(figsize=(7,4))

for sigma_s_tilde in sigma_s_tilde_list:
    K = 0.1
    for _ in range(1000):
        K = sigma_n**2 + g**2 * (-1 + (4/np.pi)*np.arctan(np.sqrt(1+np.pi*K))) 
    
    A = alpha*(sigma_s_tilde**2)/K
    B = g**2 / (1+0.5*np.pi*K)
    
    threshold = (K/np.sqrt((K-sigma_n**2)**2+sigma_s_tilde**4)) * np.sqrt(1-B**2)
    threshold_list.append(threshold)
    
    mc_theory = np.zeros(len(alpha))
    for m in range(1,100):
        mc_theory += (-1)**(m-1) * (A**m) / (1-B**m)
    
    denominator = alpha*(sigma_s_tilde**2/K)/(1-B)
    r = mc_theory/(denominator+ 1e-7)
    diverge_judge = np.append(np.where(r[1:]-r[0:-1]<0, 1, np.nan),np.nan)
    
    r_list.append(r)
    r_nondiverge_list.append(r*diverge_judge)
    

for i, sigma_s_tilde in enumerate(sigma_s_tilde_list):
    sigma_s_tilde = round(sigma_s_tilde, 2)
    plt.plot(alpha, r_list[i], color=str(1-(i+1)/10), linestyle="--")
    plt.plot(alpha, r_nondiverge_list[i], label=r"$\tilde{\sigma}_s^2 =$"+f"{sigma_s_tilde}"+r"$^2$", color=str(1-(i+1)/10))
    if int(threshold_list[i]/d_alpha) < len(alpha):
        plt.scatter(threshold_list[i], r_list[i][int(threshold_list[i]/d_alpha)], marker="*",  s=150, color=str(1-(i+1)/10))

plt.xlabel(r"$\alpha$",fontsize=20)
plt.ylabel(r"$r(\alpha)$",fontsize=20)
plt.tick_params(labelsize=13)
plt.ylim(0,1.1)    
plt.legend(fontsize=10)
plt.savefig("fig6.png")