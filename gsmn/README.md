# gsmn
The `gsmn` package contains different modules and classes for generating training data and training the GSMN. 

## Modules and Classes

* `Config_training`: This class specifies the (hyper-)parameters for the training process.
* `Control_time_strain_tension_unloading`: This class inherits from the `Control` class and defines a strain control path with tension and subsequent unloading.
* `Control`: This class defines a general strain control path. It contains attributes and methods that are not specifically tailored to a certain loading mode.
* `GSMN`: This class inherits from the `Parentmaterial` class. This class defines a GSMN. Two thermodynamic potentials must be provided to construct a GSMN. The the thermodynamic potentials can be user-defined functions (see `Potential_benchmark`) or neural networks (see `Potential_neural_network`).
* `Inout`: This module provides input and out routines, i.e., loading and saving data.
* `Parentmaterial`: This class defines a general material. It contains attributes and methods that are not specifically tailored to GSMN.
* `Plot`: This module provides plotting routines.
* `Potential_benchmark`: In this module, the thermodynamic potentials of the benchmark model are implemented. See [Flaschel et al. (2023) - Automated discovery of generalized standard material models with EUCLID](https://www.sciencedirect.com/science/article/pii/S0045782522008234).
* `Potential_neural_network`: In this module, the neural network ansatz for the thermodynamic potentials of the GSMN is implemented.