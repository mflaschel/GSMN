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
This script loads the gsmn package and creates data based on the benchmark material model.
See Flaschel et al. (2023) - Automated discovery of generalized standard material models with EUCLID.

"""

import torch

import gsmn

# ========== configuration ==========
c = gsmn.Config_training(create_outputdir=False)
torch.manual_seed(c.seed)

# ========== biaxial tests ==========
n_step = 20
epsilon_max = 0.05
noise_level = 0.0
# noise_level = 0.005

for k in range(5):
    for j in range(3):
        for i in range(2):
            if i == 0:
                time_max = 50
            elif i == 1:
                time_max = 500
            if j == 0:
                biaxial_case = "direction1"
            elif j == 1:
                biaxial_case = "direction2"
            elif j == 2:
                biaxial_case = "direction3"
            if k == 0:
                model_name = "E"
            elif k == 1:
                model_name = "VE"
            elif k == 2:
                model_name = "VEEP"
            elif k == 3:
                model_name = "EVP"
            elif k == 4:
                model_name = "VEVP"

            save_name = "data_3D_biaxial_" + model_name + "_" + biaxial_case + "_rate" + str(time_max)
            if noise_level > 0.0:
                if noise_level == 0.001:
                    save_name += "_noise1"
                if noise_level == 0.005:
                    save_name += "_noise5"
            
            # ========== define control ==========
            if biaxial_case == "direction1":
                biaxiality_factors=(1.0,0.0)
            elif biaxial_case == "direction2":
                biaxiality_factors=(1.0,1.0)
            elif biaxial_case == "direction3":
                biaxiality_factors=(0.0,1.0)

            control = gsmn.Control_time_strain_tension_unloading(
                time_max=time_max,
                epsilon_max=epsilon_max,
                n_step=n_step,
                load_case = "3D_biaxial",
                biaxiality_factors=biaxiality_factors
                )
            
            control._log = True
            control._n_log = 1
            
            # ========== define material ==========
            lim_zero = 1e-9
            lim_inf = 1e9
            if model_name == "E":
                HFEP, DRP_dual = gsmn.potential_benchmark(
                        Ginf=0.6, Kinf=1.3, G1=lim_zero, g1=lim_inf, K1=lim_zero, k1=lim_inf, sigma_0=lim_inf, eta_p=lim_zero, H_iso=lim_zero, H_kin=lim_zero
                        )
            elif model_name == "VE":
                HFEP, DRP_dual = gsmn.potential_benchmark(
                        Ginf=0.25, Kinf=0.9, G1=0.35, g1=110.0, K1=0.4, k1=15.0, sigma_0=lim_inf, eta_p=lim_zero, H_iso=lim_zero, H_kin=lim_zero
                        )
            elif model_name == "VEEP":
                HFEP, DRP_dual = gsmn.potential_benchmark(
                        Ginf=0.25, Kinf=1.3, G1=0.35, g1=110.0, K1=lim_zero, k1=lim_inf, sigma_0=0.03, eta_p=lim_zero, H_iso=0.03, H_kin=lim_zero
                        )
            elif model_name == "EVP":
                HFEP, DRP_dual = gsmn.potential_benchmark(
                        Ginf=0.6, Kinf=1.3, G1=lim_zero, g1=lim_inf, K1=lim_zero, k1=lim_inf, sigma_0=0.03, eta_p=0.04, H_iso=lim_zero, H_kin=0.01
                        )
            elif model_name == "VEVP":
                HFEP, DRP_dual = gsmn.potential_benchmark(
                        Ginf=0.25, Kinf=0.9, G1=0.35, g1=110.0, K1=0.4, k1=15.0, sigma_0=0.03, eta_p=0.04, H_iso=0.03, H_kin=0.01
                        )
            
            material = gsmn.GSMN(NNHFEP=HFEP,NNDRP_dual=DRP_dual,compute_sensitivity=False)
            material.name = model_name
            
            # ========== apply control ==========
            material.apply_control(control)
            
            # ========== add noise ==========
            if noise_level > 0.0:
                material.sigma_history += noise_level*torch.randn(material.sigma_history.shape)
            
            # ========== save ==========
            gsmn.save_biaxial_test(material, save_name, add_timestamp=False)





