# Get started
In this section we demonstrate how to preapre the environment for the experiment. 
The code is tested on windows, and in theory should also work on linux. 

## Installing related dependecies
### Using Windows with conda
Create the `pytorch-mamba` python environment based and conda and the provided enviroment.yml
``` batch
conda create -f environment.yml
coknda activate pytorch-mamba
```

## Install the local project
This provides the networks, data set, and configs shared by the expriment, which can be directly edited,

``` batch
pip install -e . 
```
