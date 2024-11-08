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
This class inherits from the Parentmaterial class.
This class defines a GSMN. Two thermodynamic potentials must be provided to construct a GSMN.
The the thermodynamic potentials can be user-defined functions (see Potential_benchmark) or neural networks (see Potential_neural_network).

"""

import numpy as np
import sys
import torch

from gsmn.Parentmaterial import Parentmaterial

class GSMN(Parentmaterial):
    
    def __init__(self,
                 NNHFEP=None,
                 # NNDRP=None,
                 NNDRP_dual=None,
                 elastic=False,
                 diff_method="AD",
                 sol_method="Newton",
                 create_graph=False,
                 compute_sensitivity=False
                 ):
        Parentmaterial.__init__(self)
        
        # check network dimensions
        if NNHFEP.n_epsilon != 6:
            raise ValueError("The Helmholtz must allow for six inputs for the strain.")
        if NNDRP_dual.n_epsilon != 0:
            raise ValueError("The dissipation potential must allow for no inputs for the strain.")
        if NNHFEP.n_internal != NNDRP_dual.n_internal:
            raise ValueError("The Helmholtz and dissipation potential must have the same number of internal variables.")
        
        # general information
        self.name = "GSMN"
        self.elastic = elastic
        
        # method
        self.diff_method = diff_method
        self.sol_method = sol_method 
        self.create_graph = create_graph 
        self.compute_sensitivity = compute_sensitivity 
        
        # potentials
        self.NNHFEP = NNHFEP
        # self.NNDRP = NNDRP
        self.NNDRP_dual = NNDRP_dual
        
        # initialize
        self.n_epsilon = NNHFEP.n_epsilon
        self.n_internal = NNHFEP.n_internal
        self.set_init()
    
    def set_init(self):
        # initialize residual
        self.R_norm_history = torch.zeros(1, dtype=torch.float64)
        
        # initialize time
        self.time_history = torch.zeros(1, dtype=torch.float64)
        
        # initialize state variables
        # 1. dimension: time history, 2. dimension: epsilon or sigma
        self.epsilon_history = torch.zeros([1,self.n_epsilon], dtype=torch.float64)
        self.sigma_history = torch.zeros([1,self.n_epsilon], dtype=torch.float64)
        
        # initialize internal variables
        self.alpha_history = torch.zeros([1,self.n_internal], dtype=torch.float64)
        
    # ========== potentials ==========
    def HFEP(self,epsilon,alpha): # Helmholtz free energy potential
        if self.NNHFEP == None:
            potential = 0.0 * epsilon + 0.0 * alpha[0]
        else:
            x = torch.cat((epsilon,alpha))
            potential = self.NNHFEP(x.view(1,-1))[0][0]
        return potential
    
    def DRP_dual(self,A_dis): # dual dissipation rate potential
        potential = 0.0
        if self.NNDRP_dual == None:
            potential += 0.0 * A_dis[0]
        else:
            potential = self.NNDRP_dual(A_dis.view(1,-1))[0][0]
        return potential
    
    # ========== automatic differentiation ==========
    def dHFEP_depsilon(self,epsilon,alpha,create_graph=False):
        # Helmholtz free energy potential dependent on epsilon
        # eliminate dependence on alpha (known)
        HFEP_fun = lambda epsilon_var : self.HFEP(epsilon_var,alpha)
        # dependent variable
        _epsilon = torch.clone(epsilon)
        # _epsilon = torch.clone(epsilon.detach())
        # _epsilon.requires_grad = True
        # derivative
        dHFEP = torch.autograd.functional.jacobian(HFEP_fun,_epsilon,create_graph=create_graph)
        return dHFEP
    
    def d2HFEP_d2epsilon(self,epsilon,alpha,create_graph=False):
        # Helmholtz free energy potential dependent on alpha
        # eliminate dependence on epsilon (known)
        HFEP_fun = lambda epsilon_var : self.HFEP(epsilon_var,alpha)
        # dependent variable
        _epsilon = torch.clone(epsilon)
        # _epsilon = torch.clone(epsilon.detach())
        # _epsilon.requires_grad = True
        # derivative
        d2HFEP = torch.autograd.functional.hessian(HFEP_fun,_epsilon,create_graph=create_graph)
        return d2HFEP
    
    def dHFEP_dalpha(self,epsilon,alpha,create_graph=False):
        # Helmholtz free energy potential dependent on alpha
        # eliminate dependence on epsilon (known)
        HFEP_fun = lambda alpha_var : self.HFEP(epsilon,alpha_var)
        # dependent variable
        _alpha = torch.clone(alpha)
        # _alpha = torch.clone(alpha.detach())
        # _alpha.requires_grad = True
        # derivative
        dHFEP = torch.autograd.functional.jacobian(HFEP_fun,_alpha,create_graph=create_graph)
        return dHFEP
    
    def d2HFEP_d2alpha(self,epsilon,alpha,create_graph=False):
        # Helmholtz free energy potential dependent on alpha
        # eliminate dependence on epsilon (known)
        HFEP_fun = lambda alpha_var : self.HFEP(epsilon,alpha_var)
        # dependent variable
        _alpha = torch.clone(alpha)
        # _alpha = torch.clone(alpha.detach())
        # _alpha.requires_grad = True
        # derivative
        d2HFEP = torch.autograd.functional.hessian(HFEP_fun,_alpha,create_graph=create_graph)
        return d2HFEP
    
    def d2HFEP_depsilon_dalpha(self,epsilon,alpha,create_graph=False):
        epsilon_alpha = torch.cat((epsilon,alpha))
        HFEP_fun = lambda epsilon_alpha : self.HFEP(epsilon_alpha[:self.n_epsilon],epsilon_alpha[self.n_epsilon:])
        d2HFEP = torch.autograd.functional.hessian(HFEP_fun,epsilon_alpha,create_graph=create_graph)
        return d2HFEP[:self.n_epsilon,self.n_epsilon:]
    
    def dDRP_dual_dA_dis(self,A_dis,create_graph=False):
        DRP_dual_fun = lambda A_dis_var : self.DRP_dual(A_dis_var)
        # dependent variable
        _A_dis = torch.clone(A_dis)
        # _A_dis = torch.clone(A_dis.detach())
        # _A_dis.requires_grad = True
        # derivative
        dDRP_dual = torch.autograd.functional.jacobian(DRP_dual_fun,_A_dis,create_graph=create_graph)
        return dDRP_dual
    
    def d2DRP_dual_d2A_dis(self,A_dis,create_graph=False):
        DRP_dual_fun = lambda A_dis_var : self.DRP_dual(A_dis_var)
        # dependent variable
        _A_dis = torch.clone(A_dis)
        # _A_dis = torch.clone(A_dis.detach())
        # _A_dis.requires_grad = True
        # derivative
        d2DRP_dual = torch.autograd.functional.hessian(DRP_dual_fun,_A_dis,create_graph=create_graph)
        return d2DRP_dual
        
    # ========== computing alpha ==========
    def solve_internal_variables(self,time_inc,epsilon,alpha_init=None,create_graph=False,compute_sensitivity=False):
        # initial guess
        if alpha_init == None:
            alpha = self.alpha_history[-1]
        else:
            alpha = alpha_init
        J = torch.eye(self.n_internal, dtype=torch.float64)
        if self.n_internal == 0 or self.elastic:
            R_norm = self.R_norm_history[-1]
        else:
            # define norm for the residual
            norm = lambda R : torch.linalg.norm(R)
            # start Newton iteration
            # ========== computing alpha using the dual dissipation potential ==========
            if self.diff_method == "AD" and self.sol_method == "Newton":
                for idx in range(self.n_Newton):
                    # Warning: it is assumed that the dual dissipation rate potential does
                    # not depend on sigma_dis
                    # derivatives
                    dHFEP = self.dHFEP_dalpha(epsilon,alpha,create_graph=create_graph)
                    d2HFEP = self.d2HFEP_d2alpha(epsilon,alpha,create_graph=create_graph)
                    dDRP_dual = self.dDRP_dual_dA_dis(-dHFEP,create_graph=create_graph)
                    d2DRP_dual = self.d2DRP_dual_d2A_dis(-dHFEP,create_graph=create_graph)
                    # Newton step for updating alpha
                    R = - alpha + self.alpha_history[-1] + time_inc*dDRP_dual
                    R_norm = norm(R)
                    if idx == 0:
                        R_init = torch.clone(R)
                        R_init_norm = norm(R_init)
                    if R_norm < self.tol_Newton:
                        # print("Converged after " + str(idx) + " iterations.")
                        break
                    J = - torch.eye(self.n_internal, dtype=torch.float64) - time_inc*torch.matmul(d2DRP_dual,d2HFEP)
                    alpha = alpha - torch.linalg.solve(J,R)
                    if np.isnan(alpha.detach()).any():
                        print("Exit Newton iteration due to nan result.")
                        sys.exit()
                    if idx == self.n_Newton-1:
                        print("Newton iteration did not converge after " + str(idx+1) + " iterations.")
                        print("Initial residual norm: " + str(R_init_norm.detach().numpy()))
                        print("Residual norm: " + str(R_norm.detach().numpy()))
                        print("Residual norm ratio: " + str(R_norm.detach().numpy() / R_init_norm.detach().numpy()))            
            # ========== computing alpha using the dual dissipation potential ==========
            
            # ========== sensitivity ==========
            if compute_sensitivity:
                J = - torch.eye(self.n_internal) - time_inc*torch.matmul(d2DRP_dual,d2HFEP) # dR / dalpha (alpha_n+1)
            # ========== sensitivity ==========
        return alpha, J, R_norm
    
    # ========== control ==========
    def time_strain_control_update(self,time_inc,epsilon_inc,alpha_init=None):
        
        create_graph = self.create_graph
        compute_sensitivity = self.compute_sensitivity
            
        # ========== time & strain control ==========
        time = self.time_history[-1] + time_inc
        epsilon = self.epsilon_history[-1] + epsilon_inc
        # ========== time & strain control ==========
        
        # ========== computing alpha ==========
        alpha, J, R_norm = self.solve_internal_variables(time_inc,epsilon,alpha_init=alpha_init,create_graph=create_graph,compute_sensitivity=compute_sensitivity)
        # ========== computing alpha ==========
        
        # ========== computing sigma ==========
        # Warning: it is assumed that the dual dissipation rate potential does
        # not depend on sigma_dis
        sigma = self.dHFEP_depsilon(epsilon,alpha,create_graph=create_graph)
        # ========== computing sigma ==========
                  
        # ========== updating history ==========
        self.time_history = torch.cat((self.time_history,time.view(1)))
        self.epsilon_history = torch.cat((self.epsilon_history,epsilon.view(1,self.n_epsilon)),dim=0)
        self.alpha_history = torch.cat((self.alpha_history,alpha.view(1,self.n_internal)),dim=0)
        self.R_norm_history = torch.cat((self.R_norm_history,R_norm.view(1)))
        self.sigma_history = torch.cat((self.sigma_history,sigma.view(1,self.n_epsilon)),dim=0)
        # ========== updating history ==========
        
    def time_mixed_control_update(self,time_inc,epsilon_inc,sigma_inc=None,load_case="3D_uniaxial"):
        
        create_graph = self.create_graph
        compute_sensitivity = self.compute_sensitivity
            
        # ========== prescribed variables and initial guesses ==========
        time = self.time_history[-1] + time_inc
        epsilon = self.epsilon_history[-1] + torch.tensor(epsilon_inc, dtype=torch.float64)
        if sigma_inc is None:
            sigma = torch.clone(self.sigma_history[-1])
        else:
            raise ValueError("Not implemented!")
            # sigma = self.sigma_history[-1] + torch.tensor(sigma_inc, dtype=torch.float64)
        alpha = self.alpha_history[-1]
        # R = torch.zeros(self.n_epsilon + self.n_internal, dtype=torch.float64)
        # J = torch.eye(self.n_epsilon + self.n_internal, dtype=torch.float64)
        # ========== prescribed variables and initial guesses ==========
        
        # ========== computing unknown variables ==========
        if load_case == "3D_uniaxial":
            # known: epsilon[0], sigma[1:]
            # unknown: sigma[0], epsilon[1:], alpha
            unit_vector = torch.zeros((self.n_epsilon,1), dtype=torch.float64)
            unit_vector[0,0] = 1.0
            zero_vector = torch.zeros((self.n_internal,1), dtype=torch.float64)
            # define norm for the residual
            norm = lambda R : torch.linalg.norm(R)
            # start Newton iteration
            if self.diff_method == "AD" and self.sol_method == "Newton":
                for idx in range(self.n_Newton):
                    # Warning: it is assumed that the dual dissipation rate potential does
                    # not depend on sigma_dis
                    # derivatives
                    dHFEP_depsilon = self.dHFEP_depsilon(epsilon,alpha,create_graph=create_graph)
                    d2HFEP_d2epsilon = self.d2HFEP_d2epsilon(epsilon,alpha,create_graph=create_graph)
                    if not (self.n_internal == 0 or self.elastic):
                        dHFEP_dalpha = self.dHFEP_dalpha(epsilon,alpha,create_graph=create_graph)
                        d2HFEP_d2alpha = self.d2HFEP_d2alpha(epsilon,alpha,create_graph=create_graph)
                        d2HFEP_depsilon_dalpha = self.d2HFEP_depsilon_dalpha(epsilon,alpha,create_graph=create_graph)
                        dDRP_dual = self.dDRP_dual_dA_dis(-dHFEP_dalpha,create_graph=create_graph)
                        d2DRP_dual = self.d2DRP_dual_d2A_dis(-dHFEP_dalpha,create_graph=create_graph)
                    if self.n_internal == 0 or self.elastic:
                        R = torch.cat((
                            dHFEP_depsilon[:1] - sigma[:1],
                            dHFEP_depsilon[1:]
                            ))
                    else:
                        R = torch.cat((
                            dHFEP_depsilon[:1] - sigma[:1],
                            dHFEP_depsilon[1:],
                            - alpha + self.alpha_history[-1] + time_inc*dDRP_dual
                            ))
                    R_norm = norm(R)
                    if idx == 0:
                        R_init = torch.clone(R)
                        R_init_norm = norm(R_init)
                    if R_norm < self.tol_Newton:
                        # print("Converged after " + str(idx) + " iterations.")
                        break
                    if self.n_internal == 0 or self.elastic:
                        J = torch.cat((
                            - unit_vector,
                            d2HFEP_d2epsilon[:,1:self.n_epsilon]
                            ), dim=1)
                        dx = torch.linalg.solve(J,R)
                        sigma[:1] = sigma[:1] - dx[:1]
                        epsilon[1:] = epsilon[1:] - dx[1:self.n_epsilon]
                    else:
                        J = torch.cat((
                            torch.cat((
                            - unit_vector,
                            d2HFEP_d2epsilon[:,1:self.n_epsilon],
                            d2HFEP_depsilon_dalpha
                            ), dim=1),
                            torch.cat((
                            zero_vector,
                            - time_inc*torch.matmul(d2DRP_dual,d2HFEP_depsilon_dalpha.T[:,1:self.n_epsilon]),
                            - torch.eye(self.n_internal, dtype=torch.float64) - time_inc*torch.matmul(d2DRP_dual,d2HFEP_d2alpha)
                            ), dim=1),
                            ), dim=0)
                        dx = torch.linalg.solve(J,R)
                        sigma[:1] = sigma[:1] - dx[:1]
                        epsilon[1:] = epsilon[1:] - dx[1:self.n_epsilon]
                        alpha = alpha - dx[self.n_epsilon:]
                    if np.isnan(alpha.detach()).any():
                        print("Exit Newton iteration due to nan result.")
                        sys.exit()
                    
                    if idx == self.n_Newton-1:
                        print("Newton iteration did not converge after " + str(idx+1) + " iterations.")
                        print("Initial residual norm: " + str(R_init_norm.detach().numpy()))
                        print("Residual norm: " + str(R_norm.detach().numpy()))
                        print("Residual norm ratio: " + str(R_norm.detach().numpy() / R_init_norm.detach().numpy()))
        elif load_case == "3D_biaxial":
            # known: epsilon[:2], sigma[2:]
            # unknown: sigma[:2], epsilon[2:], alpha
            unit_vector1 = torch.zeros((self.n_epsilon,1), dtype=torch.float64)
            unit_vector1[0,0] = 1.0
            unit_vector2 = torch.zeros((self.n_epsilon,1), dtype=torch.float64)
            unit_vector2[1,0] = 1.0
            zero_vector = torch.zeros((self.n_internal,1), dtype=torch.float64)
            # define norm for the residual
            norm = lambda R : torch.linalg.norm(R)
            # start Newton iteration
            if self.diff_method == "AD" and self.sol_method == "Newton":
                for idx in range(self.n_Newton):
                    # Warning: it is assumed that the dual dissipation rate potential does
                    # not depend on sigma_dis
                    # derivatives
                    dHFEP_depsilon = self.dHFEP_depsilon(epsilon,alpha,create_graph=create_graph)
                    d2HFEP_d2epsilon = self.d2HFEP_d2epsilon(epsilon,alpha,create_graph=create_graph)
                    if not (self.n_internal == 0 or self.elastic):
                        dHFEP_dalpha = self.dHFEP_dalpha(epsilon,alpha,create_graph=create_graph)
                        d2HFEP_d2alpha = self.d2HFEP_d2alpha(epsilon,alpha,create_graph=create_graph)
                        d2HFEP_depsilon_dalpha = self.d2HFEP_depsilon_dalpha(epsilon,alpha,create_graph=create_graph)
                        dDRP_dual = self.dDRP_dual_dA_dis(-dHFEP_dalpha,create_graph=create_graph)
                        d2DRP_dual = self.d2DRP_dual_d2A_dis(-dHFEP_dalpha,create_graph=create_graph)
                    if self.n_internal == 0 or self.elastic:
                        R = torch.cat((
                            dHFEP_depsilon[:2] - sigma[:2],
                            dHFEP_depsilon[2:]
                            ))
                    else:
                        R = torch.cat((
                            dHFEP_depsilon[:2] - sigma[:2],
                            dHFEP_depsilon[2:],
                            - alpha + self.alpha_history[-1] + time_inc*dDRP_dual
                            ))
                    R_norm = norm(R)
                    if idx == 0:
                        R_init = torch.clone(R)
                        R_init_norm = norm(R_init)
                    if R_norm < self.tol_Newton:
                        # print("Converged after " + str(idx) + " iterations.")
                        break
                    if self.n_internal == 0 or self.elastic:
                        J = torch.cat((
                            - unit_vector1,
                            - unit_vector2,
                            d2HFEP_d2epsilon[:,2:self.n_epsilon]
                            ), dim=1)
                        dx = torch.linalg.solve(J,R)
                        sigma[:2] = sigma[:2] - dx[:2]
                        epsilon[2:] = epsilon[2:] - dx[2:self.n_epsilon]
                    else:
                        J = torch.cat((
                            torch.cat((
                            - unit_vector1,
                            - unit_vector2,
                            d2HFEP_d2epsilon[:,2:self.n_epsilon],
                            d2HFEP_depsilon_dalpha
                            ), dim=1),
                            torch.cat((
                            zero_vector,
                            zero_vector,
                            - time_inc*torch.matmul(d2DRP_dual,d2HFEP_depsilon_dalpha.T[:,2:self.n_epsilon]),
                            - torch.eye(self.n_internal, dtype=torch.float64) - time_inc*torch.matmul(d2DRP_dual,d2HFEP_d2alpha)
                            ), dim=1),
                            ), dim=0)
                        dx = torch.linalg.solve(J,R)
                        sigma[:2] = sigma[:2] - dx[:2]
                        epsilon[2:] = epsilon[2:] - dx[2:self.n_epsilon]
                        alpha = alpha - dx[self.n_epsilon:]
                    if np.isnan(alpha.detach()).any():
                        print("Exit Newton iteration due to nan result.")
                        sys.exit()
                    
                    if idx == self.n_Newton-1:
                        print("Newton iteration did not converge after " + str(idx+1) + " iterations.")
                        print("Initial residual norm: " + str(R_init_norm.detach().numpy()))
                        print("Residual norm: " + str(R_norm.detach().numpy()))
                        print("Residual norm ratio: " + str(R_norm.detach().numpy() / R_init_norm.detach().numpy()))
        
        # ========== computing unknown variables ==========
        
        # ========== updating history ==========
        self.time_history = torch.cat((self.time_history,time.view(1)))
        self.epsilon_history = torch.cat((self.epsilon_history,epsilon.view(1,self.n_epsilon)),dim=0)
        self.alpha_history = torch.cat((self.alpha_history,alpha.view(1,self.n_internal)),dim=0)
        self.R_norm_history = torch.cat((self.R_norm_history,R_norm.view(1)))
        self.sigma_history = torch.cat((self.sigma_history,sigma.view(1,self.n_epsilon)),dim=0)
        # ========== updating history ==========