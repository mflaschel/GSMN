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
This class specifies the (hyper-)parameters for the training process.

"""

import argparse
import datetime
import os

from pprint import pprint

class Config_training():
    
    def __init__(self,create_outputdir=True):
        
        # ========== specify output directory ==========
        parser = argparse.ArgumentParser()
        parser.add_argument('--outputdir', default="results", type=str, help='Path to the results directory.')
        args = parser.parse_args()
        self.timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        self.outputdir = args.outputdir + "/" + self.timestamp
        if create_outputdir:
            os.makedirs(self.outputdir, exist_ok=True)

        # ========== data ==========
        
        # self.biaxial_test = [
        #     "data_3D_biaxial_E_direction1_rate50",
        #     "data_3D_biaxial_E_direction1_rate500",
        #     "data_3D_biaxial_E_direction2_rate50",
        #     "data_3D_biaxial_E_direction2_rate500",
        #     "data_3D_biaxial_E_direction3_rate50",
        #     "data_3D_biaxial_E_direction3_rate500",
        #     ]
        
        # self.biaxial_test = [
        #     "data_3D_biaxial_VE_direction1_rate50",
        #     "data_3D_biaxial_VE_direction1_rate500",
        #     "data_3D_biaxial_VE_direction2_rate50",
        #     "data_3D_biaxial_VE_direction2_rate500",
        #     "data_3D_biaxial_VE_direction3_rate50",
        #     "data_3D_biaxial_VE_direction3_rate500",
        #     ]
        
        # self.biaxial_test = [
        #     "data_3D_biaxial_EVP_direction1_rate50",
        #     "data_3D_biaxial_EVP_direction1_rate500",
        #     "data_3D_biaxial_EVP_direction2_rate50",
        #     "data_3D_biaxial_EVP_direction2_rate500",
        #     "data_3D_biaxial_EVP_direction3_rate50",
        #     "data_3D_biaxial_EVP_direction3_rate500",
        #     ]
        
        # self.biaxial_test = [
        #     "data_3D_biaxial_VEEP_direction1_rate50",
        #     "data_3D_biaxial_VEEP_direction1_rate500",
        #     "data_3D_biaxial_VEEP_direction2_rate50",
        #     "data_3D_biaxial_VEEP_direction2_rate500",
        #     "data_3D_biaxial_VEEP_direction3_rate50",
        #     "data_3D_biaxial_VEEP_direction3_rate500",
        #     ]
        
        self.biaxial_test = [
            "data_3D_biaxial_VEVP_direction1_rate50",
            "data_3D_biaxial_VEVP_direction1_rate500",
            "data_3D_biaxial_VEVP_direction2_rate50",
            "data_3D_biaxial_VEVP_direction2_rate500",
            "data_3D_biaxial_VEVP_direction3_rate50",
            "data_3D_biaxial_VEVP_direction3_rate500",
            ]
        
        self.noise_level = 0.0
        if self.noise_level == 0.005:
            for i in range(len(self.biaxial_test)):
                self.biaxial_test[i] += "_noise5"

        # ========== network architecture ==========
        self.pretrained = False
        self.pretrained_model = " "

        # ========== network architecture ==========
        self.n_epsilon = 6
        self.n_internal = 6
        self.n_neuron_HFEP = [36, 72, 36]
        # self.n_neuron_HFEP = [4, 4, 4]
        self.n_neuron_DRP = [36, 90, 180]
        # self.n_neuron_DRP = [6, 15, 30]
        self.activation = "softplus_squared"
        self.activation_alpha_init = 15.0
        self.activation_trainable = False
        self.auxiliary = "abs_smooth"
        self.auxiliary_trainable = False
        self.passthrough = "linear"
        
        # ========== differentiation method ==========
        self.use_adjoint = False
        
        # ========== optimization ==========
        self.n_iter = 1000
        self.seed = 42
        self.optimization_method = "Adam"
        self.lr = 0.01
        self.weight_decay = 1e-9
        self.use_scheduler = True # the scheduler adjusts the learning rate during training
        self.scheduler = "ReduceLROnPlateau"
        self.scheduler_factor = 2/3
        self.scheduler_patience = 5
        self.use_gradient_clipping_value = False
        self._clip_grad_value = 10000.0
        self.use_gradient_clipping_norm = True
        self._clip_grad_norm = 10000.0
        self._plot_loss = True # specify whether plots should be generated during training
        self._plot_material_response = False # specify whether plots should be generated during training
        self.n_log = self.n_iter # specify how often plots should be generated during an interactive run
        
    def _print(self):
        print('Configurations:')
        pprint(vars(self))
        print('')
        
        




