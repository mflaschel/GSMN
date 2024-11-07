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

"""

import time

from pprint import pprint

class Parentmaterial():
    
    def __init__(self):
        
        # settings
        self.n_Newton = 100
        self.tol_Newton = 1e-9
        self._log_Newton = False
        self.last_control = None
        
    def _print(self):
        # pprint(dir(self))
        pprint(vars(self))
    
    # ========== discretization ==========
    def time_discretization(self,time_inc,epsilon,epsilon_previous,method="backward_Euler"):
        # same discretization can be used for alpha
        if method == "backward_Euler":
            rate_epsilon = 1/time_inc * (epsilon - epsilon_previous)
            depsilon_drate_epsilon = time_inc
        return rate_epsilon, depsilon_drate_epsilon
    
    # ========== control ==========
    def apply_control(self,control,alpha_init=None):
        self.last_control = control
        if control._log:
            print(" ")
            print("==================================================")
            print("Apply " + control._type + " control to " + self.name + ".")
            if hasattr(self,"use_dual"):
                print("Use dual form: " + str(self.use_dual))
            if hasattr(self,"diff_method"):
                print("Differentiation method: " + self.diff_method)
            if hasattr(self,"sol_method"):
                print("Solution method: " + self.sol_method)
            start = time.time()
        for idx in range(control.n_step):
            if control._log and idx % control._n_log == 0:
                print("Load step: " + str(idx) + ".")
            if control._type == "strain":
                self.strain_control_update(control.epsilon_inc_list[idx])
            elif control._type == "time_strain":
                if control._load_case is None:
                    raise ValueError("Load case is not specified.")
                elif control._load_case == "1D" or control._load_case == "3D_zero_transversal_strain":
                    if alpha_init == None:
                        self.time_strain_control_update(control.time_inc_list[idx],control.epsilon_inc_list[idx])
                    else: # internal variables are already known   
                        self.time_strain_control_update(control.time_inc_list[idx],control.epsilon_inc_list[idx],alpha_init=alpha_init[:,idx+1])
                elif control._load_case == "3D_uniaxial" or control._load_case == "3D_biaxial":
                    self.time_mixed_control_update(control.time_inc_list[idx],control.epsilon_inc_list[idx],load_case=control._load_case)
                else:
                    raise ValueError("Load case is not implemented.")
        if control._log:
            end = time.time()
            print("End of " + control._type + " control after " + str(round(end - start,2)) + " seconds.")
            print("==================================================")
    
    def strain_control_update(self,epsilon_inc):
        print("This type of control is not supported for this material.")
    
    def time_strain_control_update(self,time_inc,epsilon_inc):
        print("This type of control is not supported for this material.")
    
    def time_mixed_control_update(self,time_inc,epsilon_inc):
        print("This type of control is not supported for this material.")

        
        
        