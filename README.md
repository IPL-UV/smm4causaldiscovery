## Pairwise Causal Discovery with Support Measure Machines

This repo contains the code to replicate the experiments in the manuscript 
`Pairwise Causal Discovery with Support Measure Machines` 


### requirements 

Experiments are run on python v3.8.12. We use the framework for pairwise causal
discovery implemented in the [The Causal Discovery
Toolbox](https://fentechsolutions.github.io/CausalDiscoveryToolbox/html/index.html).
Our method relies on kernel computed using [JAX](https://github.com/google/jax) 
and SVM available through the
[sklearn wrappers of
libsvm](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.svm). 

The complete list of required python packages is in [`requirements.txt`](requirements.txt). 

Additionally, the scripts to generate figures are written in R and requires the 
`ggplot2` and `reshape2` packages (optionally the `colorblindr``package can be used to check plots
under color blindness).


### SMM weighted ensemble 

We implement our proposed method in the `SMMwEnsemble` class available in 
`smmw_ensemble/ensemble.py`. 
An object of this class is instantiated with the following code:

```
from smmw_ensemble import SMMwEnsemble
from cdt.causality.pairwise import CDS, ANM

model = SMMwEnsemble({"CDS": CDS(), "ANM": ANM()},
                     gamma = 10)
```
