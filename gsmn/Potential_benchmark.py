"""
Author: Moritz Flaschel

      ___             ___             ___             ___     
     /\  \           /\  \           /\__\           /\__\    
    /::\  \         /::\  \         /::|  |         /::|  |   
   /:/\:\  \       /:/\ \  \       /:|:|  |        /:|:|  |   
  /:/  \:\  \     _\:\~\ \  \     /:/|:|__|__     /:/|:|  |__ 
 /:/__/_\:\__\   /\ \:\ \ \__\   /:/ |::::\__\   /:/ |:| /\__\
 \:\  /\ \/__/   \:\ \:\ \/__/   \/__/~~/:/  /   \/__|:|/:/  /
  \:\ \:\__\      \:\ \:\__\           /:/  /        |:/:/  / 
   \:\/:/  /       \:\/:/  /          /:/  /         |::/  /  
    \::/  /         \::/  /          /:/  /          /:/  /   
     \/__/           \/__/           \/__/           \/__/    

  Generalized      Standard        Material        Networks

Description:
In this module, the thermodynamic potentials of the benchmark model are implemented.
See Flaschel et al. (2023) - Automated discovery of generalized standard material models with EUCLID.

"""

import math
import torch
import torch.nn as nn

def tr(epsilon):
    return torch.sum(epsilon[0:3])

def tr_squared(epsilon):
    return tr(epsilon)**2

def vol(epsilon):
    return 1.0/3.0 * tr(epsilon) * torch.tensor([1.0, 1.0, 1.0, 0.0, 0.0, 0.0], dtype=epsilon.dtype)

def dev(epsilon):
    return epsilon - vol(epsilon)

def squared(epsilon):
    return torch.sum(epsilon**2 * torch.tensor([1.0, 1.0, 1.0, 2.0, 2.0, 2.0]))

def dev_squared(epsilon):
    return torch.sum(dev(epsilon)**2 * torch.tensor([1.0, 1.0, 1.0, 2.0, 2.0, 2.0]))

def potential_benchmark(
        Ginf=0.6, Kinf=1.3, G1=0.35, g1=110.0, K1=0.4, k1=15.0, sigma_0=0.03, eta_p=0.04, H_iso=0.03, H_kin=0.01, gamma=1e-6
        ):
    HFEP = HFEP_benchmark(Ginf=Ginf, Kinf=Kinf, G1=G1, K1=K1, H_iso=H_iso, H_kin=H_kin).double()
    DRP_dual = DRP_dual_benchmark(G1=G1, K1=K1, g1=g1, k1=k1, sigma_0=sigma_0, eta_p=eta_p, gamma=gamma).double()
    return HFEP, DRP_dual

class HFEP_benchmark(nn.Module):
    def __init__(self, Ginf=0.6, Kinf=1.3, G1=0.35, K1=0.4, H_iso=0.03, H_kin=0.01):
        super().__init__()
        self.n_epsilon = 6
        self.n_internal = 19
        self.Ginf = nn.Parameter(torch.tensor([Ginf]))
        self.Kinf = nn.Parameter(torch.tensor([Kinf]))
        self.G1 = nn.Parameter(torch.tensor([G1]))
        self.K1 = nn.Parameter(torch.tensor([K1]))
        self.H_iso = nn.Parameter(torch.tensor([H_iso]))
        self.H_kin = nn.Parameter(torch.tensor([H_kin]))
        
    def forward(self, x):
        epsilon = x[0,:self.n_epsilon]
        alpha_1 = x[0,self.n_epsilon:12]
        alpha_I = x[0,12:18]
        alpha_II = x[0,18:19]
        alpha_III = x[0,19:]
        return (
            self.Ginf * dev_squared(epsilon - alpha_I)
            + 1.0/2.0 * self.Kinf * tr_squared(epsilon - alpha_I)
            + self.G1 * dev_squared(epsilon - alpha_1 - alpha_I)
            + 1.0/2.0 * self.K1 * tr_squared(epsilon - alpha_1 - alpha_I)
            + 1.0/2.0 * self.H_iso * alpha_II**2
            + 1.0/2.0 * self.H_kin * squared(alpha_III)
            ).view(1,1)
    
class DRP_dual_benchmark(nn.Module):
    def __init__(self, G1=0.35, K1=0.4, g1=110.0, k1=15.0, sigma_0=0.03, eta_p=0.04, gamma=1e-6):
        super().__init__()
        self.n_epsilon = 0
        self.n_internal = 19
        self.G1 = nn.Parameter(torch.tensor([float(G1)]))
        self.K1 = nn.Parameter(torch.tensor([float(K1)]))
        self.g1 = nn.Parameter(torch.tensor([float(g1)]))
        self.k1 = nn.Parameter(torch.tensor([float(k1)]))
        self.sigma_0 = nn.Parameter(torch.tensor([float(sigma_0)]))
        self.eta_p = nn.Parameter(torch.tensor([float(eta_p)]))
        self.gamma = gamma
        self.beta = 1.0/self.gamma
            
    def forward(self, x):        
        A_dis_1 = x[0,:6]
        A_dis_I = x[0,6:12]
        A_dis_II = x[0,12:13]
        A_dis_III = x[0,13:]
        viscoelastic = (1/(4*self.G1*self.g1)) * dev_squared(A_dis_1) + (1/(18*self.K1*self.k1)) * tr_squared(A_dis_1)
        f = math.sqrt(3.0/2.0) * torch.norm(dev(A_dis_I+A_dis_III)) - (self.sigma_0 - A_dis_II)
        f_reg = torch.nn.functional.softplus(f, beta=self.beta)
        viscoplastic = 1.0/(2.0*self.eta_p) * torch.pow(f_reg,2.0)
        return (viscoelastic + viscoplastic).view(1,1)

