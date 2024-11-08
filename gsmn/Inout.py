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

import datetime
import numpy as np
import os
import _pickle as pickle
import sys

from gsmn.Plot import *

def save_object(objectname,filename):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    filename = filename + ".pickle"        
    pickle_out = open(filename, "wb") 
    pickle.dump(objectname,pickle_out)
    pickle_out.close()
    
def load_object(filename):
    filename = filename + ".pickle"
    pickle_in = open(filename, "rb")
    objectname = pickle.load(pickle_in)
    pickle_in.close()
    return objectname

def save_uniaxial_test(objectname,filename):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S_")
    filename = "data/uniaxial_test/" + timestamp + filename 
    save_object(objectname,filename)

def load_uniaxial_test(filename):
    filename = "data/uniaxial_test/" + filename 
    return load_object(filename)

def save_biaxial_test(objectname,filename,add_timestamp=True):
    if add_timestamp:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S_")
    else:
        timestamp = ""
    filename = "data/biaxial_test/" + timestamp + filename 
    save_object(objectname,filename)
    
def load_biaxial_test(filename):
    filename = "data/biaxial_test/" + filename 
    return load_object(filename)

def save_training_result(c=None,final=False,loss_evolution=None,GSMN=None,material_data=None):
    # ========== output directory ==========
    if c is None:
        outputdir = "results/"
        _plot_loss = False
        _plot_material_response = False
    else:
        outputdir = c.outputdir + "/"
        _plot_loss = c._plot_loss
        _plot_material_response = c._plot_material_response
    if final:
        outputdir = outputdir + "final/"
    os.makedirs(outputdir, exist_ok=True)
    
    # ========== save configurations ==========
    if c is not None:
        save_object(c,outputdir + "config_training")
        # save configurations to readable file
        original_stdout = sys.stdout
        with open(outputdir + "config_training.txt", "w") as f:
            sys.stdout = f
            c._print()
            sys.stdout = original_stdout
            
    # ========== save loss ==========
    if loss_evolution is not None:
        np.save(outputdir + "loss_evolution",loss_evolution)
        if _plot_loss:
            plot_curve((np.arange(len(loss_evolution)),loss_evolution),axislabels=("Iterations","Loss"),legendentries=("loss"," "),savepath=outputdir + "loss_evolution")
            plot_curve_semilogy((np.arange(len(loss_evolution)),loss_evolution),axislabels=("Iterations","Loss"),legendentries=("loss"," "),savepath=outputdir + "loss_evolution_log")
    
    # ========== save neural network ==========
    if GSMN is not None:
        save_object(GSMN,outputdir + "GSMN")
    
    # ========== save material data ==========
        if material_data is not None:
            save_object(material_data,outputdir + "material_data")
    
    






















