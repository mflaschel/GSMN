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

import matplotlib.pyplot as plt
import numpy as np
import time
import torch

plt.rcParams['font.size'] = 14  # Global default font size
plt.rcParams['axes.titlesize'] = 14  # Title font size for axes
plt.rcParams['axes.labelsize'] = 14  # Axis labels font size
plt.rcParams['xtick.labelsize'] = 14  # X-axis tick labels font size
plt.rcParams['ytick.labelsize'] = 14  # Y-axis tick labels font size
plt.rcParams['legend.fontsize'] = 14  # Legend font size
plt.rcParams['figure.titlesize'] = 14  # Font size for figure title

_FORMAT = "png"

def plot_curve(data0,data1=None,axislabels=None,style=('-b','--r'),legendentries=None,savepath=None):
    # plt.figure()
    if legendentries is None:
        legendentries = (" "," ")
    plt.plot(data0[0],data0[1],style[0],label=legendentries[0])
    if data1 is not None:
        plt.plot(data1[0],data1[1],style[1],label=legendentries[1])
    if axislabels is not None:
        plt.xlabel(axislabels[0])
        plt.ylabel(axislabels[1])
    plt.legend()
    if savepath is not None:
        savepath = savepath + "." + _FORMAT
        plt.tight_layout()
        plt.savefig(savepath, format=_FORMAT)
        plt.close('all')
    else:
        plt.show()
        
def plot_curve_semilogy(data0,data1=None,axislabels=None,style=('-b','--r'),legendentries=None,savepath=None):
    # plt.figure()
    if legendentries is None:
        legendentries = (" "," ")
    plt.semilogy(data0[0],data0[1],style[0],label=legendentries[0])
    if data1 is not None:
        plt.semilogy(data1[0],data1[1],style[1],label=legendentries[1])
    if axislabels is not None:
        plt.xlabel(axislabels[0])
        plt.ylabel(axislabels[1])
    plt.legend()
    if savepath is not None:
        savepath = savepath + "." + _FORMAT
        plt.tight_layout()
        plt.savefig(savepath, format=_FORMAT)
        plt.close('all')
    else:
        plt.show()





