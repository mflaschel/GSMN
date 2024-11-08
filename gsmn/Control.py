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
This class defines a general strain control path. It contains attributes and methods that are not specifically tailored to a certain loading mode.

"""

import numpy as np
import sys
import torch

from pprint import pprint

class Control():
    
    def __init__(self):
        
        self._type = None
        self._load_case = None
        self.n_step = None
        self.time_inc_list = None
        self.epsilon_inc_list = None
        
        self._log = False
        self._n_log = 10
    
    def _print(self):
        # pprint(dir(self))
        pprint(vars(self))
        
    def read_data(self,data):
        if hasattr(data, 'time_history') and hasattr(data, 'epsilon_history'):
            self._type = "time_strain"
            if data.last_control is not None:
                self._load_case = data.last_control._load_case
            time_history = data.time_history
            epsilon_history = data.epsilon_history
            if torch.is_tensor(time_history):
                time_history = time_history.detach().numpy()
            if torch.is_tensor(epsilon_history):
                epsilon_history = epsilon_history.detach().numpy()
            self.time_inc_list = np.diff(time_history)
            self.epsilon_inc_list = np.diff(epsilon_history,axis=0)
            self.n_step = len(self.time_inc_list)
        else:
            print("Warning in Control.py: Reading this type of control from data is not implemented.")
            print("Hint: Does the dataset contain information about the time?")
            sys.exit()
        