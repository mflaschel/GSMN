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
In this module, the neural network ansatz for the thermodynamic potentials of the GSMN is implemented.

"""

import math
import torch
import torch.nn as nn

from torch import Tensor
from torch.nn.parameter import Parameter 

# ========== print model information ==========
def print_torch_architecture(network):
    print("Network modules:")
    for i in network.named_modules():
        if i[0] != '':
            print(i)
    print(" ")

def print_count_torch_parameters(network):
    n_parameters_total = 0
    print("Number of trainable parameters:")
    for name, param in network.named_parameters():
        print(name + ":", param.numel(), "parameters")
        n_parameters_total += param.numel()
    print("Total number of trainable parameters: ", str(n_parameters_total))
    print(" ")
    return

def print_torch_parameters(network, extend = False):
    print("Network parameters:")
    for name, param in network.named_parameters():
        if extend:
            print(name, "is a", param)
            print(" ")
        else:
            print(name + ":", param.data)
    if not extend:
        print(" ")
    return

# ========== managing parameters and gradients ==========
def count_torch_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def get_torch_parameters(model):
    return torch.cat([p.view(-1) for p in model.parameters()])
   
def set_torch_parameters(model, theta):
    idx = 0
    for p in model.parameters():
        p.data = theta[idx:idx+p.numel()].view_as(p)
        idx += p.numel()
    return model

def set_zero_torch_parameters(model):
    for p in model.parameters():
        p.data = torch.zeros_like(p)
    return model

def detach_torch_parameters(model):
    for p in model.parameters():
        p.detach_() # the "_" specifies in-place detachment
        
def overwrite_torch_parameters(model, theta):
    idx = 0
    for p in model.parameters():
        p = theta[idx:idx+p.numel()].view_as(p)
        idx += p.numel()
    return model

def get_torch_grad(model):
    return torch.cat([p.grad.view(-1) for p in model.parameters()])

def set_torch_grad(model, grad_theta):
    idx = 0
    for p in model.parameters():
        p.grad = grad_theta[idx:idx+p.numel()].view_as(p)
        idx += p.numel()
    return model

# ========== activation functions and auxiliary functions ==========
class softplus(nn.Module):
    def __init__(self, alpha_init = 15.0, trainable = True):
        super(softplus,self).__init__()
        
        self.trainable = trainable

        # initialize alpha
        if trainable:
            self.alpha = Parameter(torch.tensor(alpha_init))
        else:
            self.alpha = alpha_init
        
    def forward(self, x):
        beta = self.alpha**2
        if self.trainable:
            return torch.log(1 + torch.exp(beta * x)) / beta
        else:
            return torch.nn.functional.softplus(x, beta=beta)

class softplus_squared(nn.Module):
    def __init__(self, alpha_init = 15.0, trainable = True):
        super(softplus_squared,self).__init__()
        
        self.trainable = trainable

        # initialize alpha
        if trainable:
            self.alpha = Parameter(torch.tensor(alpha_init))
        else:
            self.alpha = alpha_init
                        
    def forward(self, x):
        beta = self.alpha**2
        if self.trainable:
            return torch.square( torch.log(1 + torch.exp(beta * x)) / beta ) 
        else:
            return torch.square( torch.nn.functional.softplus(x, beta=beta) )
        
class abs_smooth(nn.Module):
    def __init__(self, alpha_init = 0.1, trainable = True):
        super(abs_smooth,self).__init__()
        
        self.trainable = trainable

        # initialize alpha
        if trainable:
            self.alpha = Parameter(torch.tensor(alpha_init))
        else:
            self.alpha = alpha_init
        
    def forward(self, x):
        beta = self.alpha**2
        return torch.sqrt(torch.pow(x,2) + beta) - beta**(0.5)

# ========== network layers ==========
class passthrough_quadratic(nn.Module):
    def __init__(self, in_features, n_neuron, A = None):
        super(passthrough_quadratic,self).__init__()
        self.in_features = in_features

        if A == None:
            self.A = Parameter(torch.zeros([n_neuron,in_features]))
        else:
            self.A = Parameter(torch.tensor(A))
            
    def forward(self, x):
        Ax = nn.functional.linear(x, self.A, None)
        xAAx = torch.diag(torch.tensordot(Ax,Ax,dims=([1],[1]))).view([-1, 1]) # includes unnecessary computations
        return xAAx
    
class LinearPosWeights(nn.Module):
    r"""Applies a linear transformation to the incoming data: :math:`y = xA^T + b`

    This module supports :ref:`TensorFloat32<tf32_on_ampere>`.

    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        bias: If set to ``False``, the layer will not learn an additive bias.
            Default: ``True``

    Shape:
        - Input: :math:`(*, H_{in})` where :math:`*` means any number of
          dimensions including none and :math:`H_{in} = \text{in\_features}`.
        - Output: :math:`(*, H_{out})` where all but the last dimension
          are the same shape as the input and :math:`H_{out} = \text{out\_features}`.

    Attributes:
        weight: the learnable weights of the module of shape
            :math:`(\text{out\_features}, \text{in\_features})`. The values are
            initialized from :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})`, where
            :math:`k = \frac{1}{\text{in\_features}}`
        bias:   the learnable bias of the module of shape :math:`(\text{out\_features})`.
                If :attr:`bias` is ``True``, the values are initialized from
                :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})` where
                :math:`k = \frac{1}{\text{in\_features}}`

    Examples::

        >>> m = nn.Linear(20, 30)
        >>> input = torch.randn(128, 20)
        >>> output = m(input)
        >>> print(output.size())
        torch.Size([128, 30])
    """
    __constants__ = ['in_features', 'out_features']
    in_features: int
    out_features: int
    weight: Tensor

    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 auxiliary: str = "softplus", trainable: bool = True,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(LinearPosWeights, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.empty((out_features, in_features), **factory_kwargs))
        if bias:
            self.bias = Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
        
        # auxiliary function
        if auxiliary == "softplus":
            self.auxiliary = softplus(trainable = trainable)
        elif auxiliary == "abs_smooth":
            self.auxiliary = abs_smooth(trainable = trainable)
        else:
            raise ValueError("Not implemented.")

    def reset_parameters(self) -> None:
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(in_features), 1/sqrt(in_features)). For details, see
        # https://github.com/pytorch/pytorch/issues/57109
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input: Tensor) -> Tensor:
        self.pos_weights = self.auxiliary(self.weight)
        return nn.functional.linear(input, self.pos_weights, self.bias)

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )

# ========== network architecture ==========
class Potential_neural_network(nn.Module):
    def __init__(self,
                 n_epsilon = 1,
                 n_internal = 1,
                 n_neuron = [1],
                 activation = "softplus_squared",
                 activation_alpha_init = 15.0,
                 activation_trainable = False,
                 auxiliary = "abs_smooth",
                 auxiliary_trainable = False,
                 passthrough = "linear"
                 ):
        super().__init__()
        self.n_epsilon = n_epsilon
        self.n_internal = n_internal
        
        n_input = n_epsilon + n_internal # strain and internal variables
        n_output = 1
        
        self.n_layer = len(n_neuron)
        self.layer = nn.ModuleDict()
        self.passthroughlayer = nn.ModuleDict()
        self.layer[str(0)] = torch.nn.Linear(n_input,n_neuron[0],bias=True)
        for i in range(1, self.n_layer):
            self.layer[str(i)] = LinearPosWeights(n_neuron[i-1],n_neuron[i],bias=True,auxiliary=auxiliary,trainable=auxiliary_trainable)
            self.passthroughlayer[str(i)] = torch.nn.Linear(n_input,n_neuron[i],bias=False)
        self.layer[str(self.n_layer)] = LinearPosWeights(n_neuron[self.n_layer-1],n_output,bias=True,auxiliary=auxiliary,trainable=auxiliary_trainable)
        if passthrough == "linear":
            self.passthroughlayer[str(self.n_layer)] = torch.nn.Linear(n_input,n_output,bias=False)
        elif passthrough == "quadratic":
            self.passthroughlayer[str(self.n_layer)] = passthrough_quadratic(n_input,n_input)
        else:
            raise ValueError("Not implemented")
        
        # activation function
        self.activation = nn.ModuleDict()
        if activation == "softplus":
            for i in range(0, self.n_layer):
                self.activation[str(i)] = softplus(alpha_init=activation_alpha_init,trainable=activation_trainable)
        elif activation == "softplus_squared":
            for i in range(0, self.n_layer):
                self.activation[str(i)] = softplus_squared(alpha_init=activation_alpha_init,trainable=activation_trainable)
        else:
            raise ValueError("Not implemented")

    def forward_convex(self, x):
        x0 = torch.clone(x)
        x = self.activation[str(0)](self.layer[str(0)](x))
        for i in range(1,self.n_layer):
            x = self.activation[str(i)]( self.layer[str(i)](x) + self.passthroughlayer[str(i)](x0) )
        x = self.layer[str(self.n_layer)](x) + self.passthroughlayer[str(self.n_layer)](x0)
        return x
    
    def forward(self, x):
        x_zeros = torch.zeros_like(x)
        
        x_var = torch.zeros_like(x, requires_grad=True)
        forward_convex = self.forward_convex(x_var)
        dforward_convex = torch.autograd.grad(forward_convex, x_var, create_graph=True)[0]
        # the latter is only possible if forward_convex is scalar, i.e., not for vectorized predictions
        
        x = self.forward_convex(x) - torch.dot(dforward_convex[0],x[0]).view(1,1) - self.forward_convex(x_zeros)
        return x
        
        
    