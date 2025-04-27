import numpy as np
from tqdm import tqdm
import scipy.stats as sps
import sys
import torch

def memory_calc(g,N,act_func,leadout,sigma, trial=10, Tobs= 10000, Tinit=1000, max_delay=500):
    memorys = np.zeros([trial, max_delay]) 
    for i in range(trial):
        J = np.random.normal(0,g/np.sqrt(N),[N,N])
        v = np.random.normal(0,1,N)

        x = np.zeros([N,Tinit+Tobs])
        inputs = np.random.normal(0,sigma,Tinit+Tobs)

        for t in tqdm(range(Tinit+Tobs-1)):
            x[:,t+1] = J@act_func(x[:,t]) + v*inputs[t+1] 
        
        kernel = np.linalg.inv((x[:leadout,Tinit:]@x[:leadout,Tinit:].T)/Tobs)
        
        eps = sps.chi2.ppf(q = 1-1e-4, df = leadout)*2/Tobs #threshold of Capacity
            
        for d in range(max_delay):
            a = np.dot(x[:leadout,Tinit:], inputs[Tinit-d : Tinit+Tobs-d])/Tobs
            md = np.dot(a, np.dot(kernel, a))/(sigma**2)
            memorys[i, d] = md*(md>eps)
        print(f"trial={i+1}/{trial} was done") 
    return np.mean(memorys, axis=0), np.std(memorys, axis=0)  
    
def mc_calc(g,N,act_func,max_leadout,sigma, trial=10, Tobs= 10000, Tinit=1000, max_delay=500, device="mps", sigma_noise=0.0, dinit=0):
    mcs = np.zeros([trial, max_leadout]) 
    dev = torch.device(device)
    
    for i in range(trial):
        J = ((g/np.sqrt(N))*torch.randn(N,N)).to(dev)
        v = torch.randn(N).to(dev)

        x = (torch.zeros([N,Tinit+Tobs])).to(dev)
        inputs = (sigma*torch.randn(Tinit+Tobs)).to(dev) 
        noise = (sigma_noise*torch.randn(N, Tinit+Tobs)).to(dev)

        for t in tqdm(range(Tinit+Tobs-1)):
            x[:,t+1] = J@act_func(x[:,t]) + v*inputs[t+1] + noise[:, t+1]

        x = x.to("cpu").numpy()
        inputs = inputs.to("cpu").numpy()
        
    
        for l in range(max_leadout+1):
            kernel = np.linalg.inv((x[:l,Tinit:]@x[:l,Tinit:].T)/Tobs)
            eps = sps.chi2.ppf(q = 1-1e-4, df = l)*2/Tobs #threshold of Capacity
            
            memory = []
            for d in range(dinit,max_delay):
                a = np.dot(x[:l,Tinit:], inputs[Tinit-d : Tinit+Tobs-d])/Tobs
                md = np.dot(a, np.dot(kernel, a))/(sigma**2)
                memory.append(md*(md>eps))
            mcs[i,l-1] = np.sum(memory) 
        print(f"trial={i+1}/{trial} was done") 
    return np.mean(mcs, axis=0), np.std(mcs, axis=0)  

        
def cor_calc(g,N,act_func, sigma, sigma_noise=0.0, trial=10, Tobs= 10000, Tinit=1000, device="mps"):
    cor_list = []
    dev = torch.device(device)
    for _ in range(trial):
        J = ((g/np.sqrt(N))*torch.randn(N,N)).to(dev)
        v = torch.randn(N).to(dev)

        x = (torch.zeros([N,Tinit+Tobs])).to(dev)
        inputs = (sigma*torch.randn(Tinit+Tobs)).to(dev) 
        noise = (sigma_noise*torch.randn(N, Tinit+Tobs)).to(dev)

        for t in range(Tinit+Tobs-1):
            x[:,t+1] = J@act_func(x[:,t]) + v*inputs[t+1] + noise[:, t+1]

        x = x.to("cpu").numpy()
        
        C = x[:,Tinit:]@(x[:,Tinit:].T)/Tobs
        R = C/np.mean(np.diag(C))
        Rabs = np.abs(R)
        Rabs[range(N), range(N)] = np.nan
        cor_list.append(np.nanmean(Rabs))
    
    return np.mean(cor_list)