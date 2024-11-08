# Generalized Standard Material Networks
```
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
```
Generalized Standard Material Networks (GSMN) constitute a general machine learning framework based on convex neural networks for learning the mechanical behavior of generalized standard materials. The modules and classes for generating training data and training the GSMN are implemented in the `gsmn` package, which can be imported with `import gsmn`.

To run the code, execute one of the following scripts:
* `main_biaxial_test_3D.py`: This script loads the gsmn package and creates data based on the benchmark material model. See [Flaschel et al. (2023) - Automated discovery of generalized standard material models with EUCLID](https://www.sciencedirect.com/science/article/pii/S0045782522008234).
* `main_training_biaxial_multiple.py`: This script loads the gsmn package and trains the GSMN on the training data.

## Requirements

* `matplotlib`
* `numpy`
* `torch`