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
Generalized Standard Material Networks (GSMN) constitute a general machine learning framework based on convex neural networks for learning the mechanical behavior of generalized standard materials, see [publication](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5023581). The modules and classes for generating training data and training the GSMN are implemented in the `gsmn` package, which can be imported with `import gsmn`.

To run the code, execute one of the following scripts:
* `main_biaxial_test_3D.py`: This script loads the `gsmn` package and creates data based on the benchmark material model. See [Flaschel et al. (2023) - Automated discovery of generalized standard material models with EUCLID](https://www.sciencedirect.com/science/article/pii/S0045782522008234).
* `main_training_biaxial_multiple.py`: This script loads the `gsmn` package and trains the GSMN on the training data.

## Requirements

* `matplotlib`
* `numpy`
* `torch`

## How to cite the code

```
Moritz Flaschel, Paul Steinmann, Laura De Lorenzis and Ellen Kuhl  
Supplementary software for ”Convex neural networks learn generalized standard material models”  
2024  
DOI: http://doi.org/10.5281/zenodo.14055700  
```
