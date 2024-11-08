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

import numpy as np
import time
import torch

import gsmn

# ========== configuration ==========
c = gsmn.Config_training()
c._print()
torch.manual_seed(c.seed)

# ========== load data ==========
n_datasets = len(c.biaxial_test)
material_data = {}
control_information = {}
for dataset in range(n_datasets):
    material_data[str(dataset)] = gsmn.load_biaxial_test(c.biaxial_test[dataset])
    control_information[str(dataset)] = gsmn.Control()
    control_information[str(dataset)].read_data(material_data[str(dataset)])

# ========== define networks ==========
if c. pretrained:
    GSMN = gsmn.load_object(c.pretrained_model)
    NNHFEP = GSMN.NNHFEP
    NNDRP_dual = GSMN.NNDRP_dual
else:
    NNHFEP = gsmn.Potential_neural_network(
        n_epsilon = 6,
        n_internal = c.n_internal,
        n_neuron = c.n_neuron_HFEP,
        activation = c.activation,
        activation_alpha_init = c.activation_alpha_init,
        activation_trainable = c.activation_trainable,
        auxiliary = c.auxiliary,
        auxiliary_trainable = c.auxiliary_trainable,
        passthrough = c.passthrough
        ).double()
    
    NNDRP_dual = gsmn.Potential_neural_network(
        n_epsilon = 0,
        n_internal = c.n_internal,
        n_neuron = c.n_neuron_DRP,
        activation = c.activation,
        activation_alpha_init = c.activation_alpha_init,
        activation_trainable = c.activation_trainable,
        auxiliary = c.auxiliary,
        auxiliary_trainable = c.auxiliary_trainable,
        passthrough = c.passthrough
        ).double()

# # ========== optimizer and loss function ==========
# SGD means stochastic gradient descent
# ADAM means adaptive moment estimation
# lr means learning rate
if c.optimization_method == "SGD":
    optimizer_NNHFEP = torch.optim.SGD(NNHFEP.parameters(), lr=c.lr, weight_decay=c.weight_decay)
    optimizer_NNDRP_dual = torch.optim.SGD(NNDRP_dual.parameters(), lr=c.lr, weight_decay=c.weight_decay)
if c.optimization_method == "Adam":
    optimizer_NNHFEP = torch.optim.Adam(NNHFEP.parameters(), lr=c.lr, weight_decay=c.weight_decay)
    optimizer_NNDRP_dual = torch.optim.Adam(NNDRP_dual.parameters(), lr=c.lr, weight_decay=c.weight_decay)
loss_func = torch.nn.MSELoss()

if c.use_scheduler:
    if c.scheduler == "ReduceLROnPlateau":
        scheduler_NNHFEP = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_NNHFEP, mode='min', factor=c.scheduler_factor, patience=c.scheduler_patience)  
        scheduler_NNDRP_dual = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_NNDRP_dual, mode='min', factor=c.scheduler_factor, patience=c.scheduler_patience)  
    else:
        raise ValueError("Not implemented.")
    
# ========== training ==========
target = torch.zeros(0)
for dataset in range(n_datasets):
    target = torch.cat((target, material_data[str(dataset)].sigma_history[1:,0], material_data[str(dataset)].sigma_history[1:,1]))
if type(target).__module__ == np.__name__:
    target = torch.from_numpy(target)
else:
    target = target

# loss
loss_evolution = np.array([])
loss_min = np.inf

start = time.time()
for i in range(c.n_iter + 1):
    print()
    print("Step: " + str(i))
        
    # prediction
    if not c.use_adjoint:
        GSMN_current = gsmn.GSMN(NNHFEP=NNHFEP,NNDRP_dual=NNDRP_dual,create_graph=True)
    else:
        raise RuntimeError("This is not yet implemented. Please set use_adjoint=False.")

    prediction = torch.zeros(0)
    for dataset in range(n_datasets):
        GSMN_current.set_init()
        GSMN_current.apply_control(control_information[str(dataset)])
        prediction = torch.cat((prediction, GSMN_current.sigma_history[1:,0], GSMN_current.sigma_history[1:,1]))
    
    if c.use_scheduler:
        str_lr = "Learning rate:"
        for param_group in optimizer_NNHFEP.param_groups:
            str_lr += " " + str(param_group['lr'])
        for param_group in optimizer_NNDRP_dual.param_groups:
            str_lr += " " + str(param_group['lr'])
        print(str_lr)
    
    # compute loss
    loss = loss_func(prediction, target)
    if i == 0:
        print("Loss: " + str(loss.data.numpy()))
    else:
        print("Loss: " + str(loss.data.numpy()) + ", Loss change: " + str(loss.data.numpy() - loss_evolution[-1]))        
    if loss.data.numpy() < loss_min:
        print("New minimum found. Save results...")
        loss_min = loss.data.numpy()
        gsmn.save_training_result(c=c,final=False,loss_evolution=loss_evolution,GSMN=GSMN_current,material_data=material_data[str(dataset)])
        
    loss_evolution = np.append(loss_evolution,loss.data.numpy())
    if np.isnan(loss.data.numpy()).any():
        raise RuntimeError("Exit training due to nan loss.")
    optimizer_NNHFEP.zero_grad()
    optimizer_NNDRP_dual.zero_grad()
    
    # compute gradient
    if not c.use_adjoint:
        loss.backward()
    
    # apply optimization step
    if c.use_gradient_clipping_value:
        torch.nn.utils.clip_grad_value_(NNHFEP.parameters(), clip_value=c._clip_grad_value)
        torch.nn.utils.clip_grad_value_(NNDRP_dual.parameters(), clip_value=c._clip_grad_value)
    if c.use_gradient_clipping_norm:
        torch.nn.utils.clip_grad_norm_(NNHFEP.parameters(), c._clip_grad_norm)
        torch.nn.utils.clip_grad_norm_(NNDRP_dual.parameters(), c._clip_grad_norm)
    optimizer_NNHFEP.step()
    optimizer_NNDRP_dual.step()
    if c.use_scheduler:
        scheduler_NNHFEP.step(loss)
        scheduler_NNDRP_dual.step(loss)
            
    # save model
    if i == c.n_iter:
        gsmn.save_training_result(c=c,final=True,loss_evolution=loss_evolution,GSMN=GSMN_current,material_data=material_data[str(0)])

end = time.time()
print("Time needed for training: " + str(round(end - start,2)) + " seconds")




