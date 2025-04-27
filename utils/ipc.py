import numpy as np
import math
from scipy.special import  eval_hermitenorm, eval_legendre
from tqdm import tqdm
from scipy.linalg import pinv
import scipy.stats as sps

def gen_degs(degsum,num):
    if num==1:
        return [[degsum]]
    gen_list = []
    for i in range(1,degsum-num+2):
        list_ = gen_degs(degsum-i,num-1)
        for j in range(len(list_)):
            list_[j].append(i)
        gen_list += list_
    return gen_list


# Enumerate all narrowly monotonically increasing sequences totaling num that start with 0 and end with window
# but num>=2、window >= num-1
def gen_delays(window,num):
    if num==2:
        return [[0,window]]
    gen_list = []
    for i in range(num-2,window):
        list_ = gen_delays(i,num-1)
        for j in range(len(list_)):
           list_[j].append(window)
        gen_list += list_
    return gen_list

# Given a signal sequence x (number of units x time) of reservoirs, an inverse matrix xx_inv, a teacher time series z, and a cutoff value eps, find capacity C_ε
def calc_Ceps(z,x,xx_inv,eps):
    zx = x@z/len(z)
    z2 = np.dot(z,z)/len(z)
    C_T = ((zx.T)@xx_inv@zx)/z2
    return C_T * (C_T > eps) 

# Return the value of the normalized Hermite polynomial
def norm_hermite(deg, input):
    return  eval_hermitenorm(deg,input)/np.sqrt(math.factorial(deg))

def norm_Legendre(deg, input):
    return np.sqrt((2/(2*deg+1))) * eval_legendre(deg,input)

    
"""
Calculation of IPC
"""
def calc_ipc(N, g, input_intensity, max_leadout, degree_list, Tobs, Tinit, act_func, noise_std=0.0):

    J = np.random.normal(0, g/np.sqrt(N), [N,N]) #recurrent weights of a reservoir
    v = np.random.normal(0, 1, N) #input weights

    L_list = np.arange(1,max_leadout+1) #list of the num of readout neurons
    
    ipc = np.zeros([len(L_list),len(degree_list)])

    x = np.zeros([N,Tinit+Tobs])
    x[:,0] = np.random.normal(0,1,N)
    input = np.random.normal(0, 1, Tinit+Tobs) 
    noise = np.random.normal(0,noise_std, [N,Tinit+Tobs])

    for t in tqdm(range(1, Tinit+Tobs)):
        x[:,t] = J@act_func(x[:,t-1]) + input[t-1]*v*input_intensity + noise[:,t]

    for idL in tqdm(range(len(L_list))):
        L = L_list[idL]
        eps = sps.chi2.ppf(q = 1-1e-4, df = L)*2/Tobs #閾値の決定

        cov = x[:L, Tinit:]@x[:L, Tinit:].T/Tobs
        x_inv  = np.linalg.inv(cov)

        for deg_id in range(len(degree_list)):
            degree = degree_list[deg_id]
            C_deg = 0

            if degree>=7:
                MAX_WINDOW = 10
            else:
                MAX_WINDOW = 100

            for num_deg in range(1,degree+1):
                deg_list = gen_degs(degree, num_deg)
                jdg = 1
                C_numdeg = 0

                for degs in deg_list:
                    C_maxwindow = 0
                    for window in range(num_deg-1,MAX_WINDOW):
                        if jdg ==0:
                            break 
                        if num_deg >= 2:
                            rel_delay_list = gen_delays(window,num_deg)
                        else:
                            rel_delay_list = [[0]]
                            jdg -= 1

                        C_mindelay = 0
                        for rel_delay in rel_delay_list:
                            rel_delay = np.array(rel_delay)

                            for mindelay in range(1,10000):
                                delay = rel_delay + mindelay
                                
                                # Generate teacher time series z(t)
                                z = np.ones(Tobs)
                                for n in range(num_deg):
                                    z *= norm_hermite(degs[n],input[Tinit-delay[n]:Tobs+Tinit-delay[n]])
                                z = z.reshape([Tobs,1])
                                zx = (x[:L, Tinit:]@z)/Tobs
                                C_T = (zx.T@(x_inv@zx))[0,0]/np.mean(z**2)
                                Ceps = C_T*(C_T>eps)
                                C_mindelay += Ceps
                                if Ceps==0 and mindelay>5:
                                    break

                        C_maxwindow += C_mindelay
                        if C_mindelay == 0:
                            break

                    C_numdeg += C_maxwindow
                
                C_deg += C_numdeg

            ipc[idL,deg_id] = C_deg
    
    return ipc
