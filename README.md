# ABC-NET
This package allows to compute the Intrinsic Dimension of unweighted networks using the I3D estimator





, contained in the [DADApy package]) (reference [here](https://www.sciencedirect.com/science/article/pii/S2666389922002070)).

Parameter inference and model selection for network generating models



## Installation guide
The routines in the package are mainly based on [graph-tool](https://github.com/antmd/graph-tool), [DADApy](https://github.com/sissa-data-science/DADApy) and [PyABC](https://github.com/ICB-DCM/pyABC/). In order to have everything working properly we suggest to install graph-tool first:
```
conda create --name gt -c conda-forge graph-tool
conda activate gt
```
Look [here](https://graph-tool.skewed.de/static/doc/index.html#installing-graph-tool) for alternative installations.

Then it you can install PyABC and DADApy through pip:
```
pip install pyabc
pip install dadapy
```
If you want to be sure to get the latest releases:
```
pip install git+https://github.com/icb-dcm/pyabc.git
pip install git+https://github.com/sissa-data-science/DADApy
```
A couple of routine are based on NetworkX, so we recommend to install it too, always through pip.

Refer to [DADApy documentation](https://dadapy.readthedocs.io/en/latest/index.html) in order to have mode details about the ID estimator for discrete spaces.

## What's in the folder
 -  The module with all the needed functions to generate the simplest graphs
 -  The module to extract the Intrinsic Dimension and other observables from the graphs
 -  The module to perform SMC-ABC
 -  C++ code for ID guided ABC
 -  Notebook examples:
  - ID calculation
  - SMC-ABC
