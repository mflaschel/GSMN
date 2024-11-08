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
This class inherits from the Control class and defines a strain control path with tension and subsequent unloading.

"""

import numpy as np

from gsmn.Control import Control

class Control_time_strain_tension_unloading(Control):
    
    def __init__(self,time_max=1.0,epsilon_max=1.0,n_step=100,load_case="1D",biaxiality_factors=(1.0,0.0)):
        Control.__init__(self)
        
        self._type = "time_strain"
        self._load_case = load_case
        n_step_tension = n_step
        n_step_compression = n_step
        self.n_step = n_step_tension + n_step_compression
        
        time_inc = time_max / n_step_tension
        self.time_inc_list = time_inc*np.ones(self.n_step)
        
        epsilon_inc = epsilon_max / n_step_tension
        epsilon_inc_list = np.append(epsilon_inc*np.ones(n_step_tension),-epsilon_inc*np.ones(n_step_compression))
        
        if load_case == "1D":
            self.epsilon_inc_list = epsilon_inc_list
        
        elif load_case == "3D_zero_transversal_strain" or load_case == "3D_uniaxial":
            self.epsilon_inc_list = np.zeros((len(epsilon_inc_list),6))
            self.epsilon_inc_list[:,0] = epsilon_inc_list
        
        elif load_case == "3D_biaxial":
            self.epsilon_inc_list = np.zeros((len(epsilon_inc_list),6))
            self.epsilon_inc_list[:,0] = epsilon_inc_list * biaxiality_factors[0]
            self.epsilon_inc_list[:,1] = epsilon_inc_list * biaxiality_factors[1]
