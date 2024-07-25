# SaccpaNet: A Separable Atrous Convolution-based Cascade Pyramid Attention Network to Estimate Body Landmarks Using Cross-modal Knowledge Transfer for Under-blanket Sleep Posture Classification 


# Get started
In this section we demonstrate how to preapre the environment for the experiment. 
The code is tested on windows, and in theory should also work on linux. 

## Installing related dependecies
### Using Windows with conda
Create the `pytorch-mamba` python environment based and conda and the provided enviroment.yml
``` batch
conda env create -f environment.yml
conda activate posture_experiment
```
### Building the environment from scratch with conda and pip
Please be aware that pytorch-lightning and sqlsalchemy are setted to specific version as newer versions introduces breaking changes. 
``` batch
conda create -n posture_experiment -c pytorch -c nvidia -c conda-forge -c default python=3.9 pytorch torchvision torchaudio pytorch-cuda pandas seaborn -y
conda activate posture_experiment
pip install pytorch-lightning==1.6.4 SQLAlchemy==1.4.48 opencv-python scikit-learn tqdm torchmetrics
```


## Install the local project
This provides the networks, data set, and configs shared by the expriment, which can be directly edited,

``` batch
pip install -e . 
```

## installing dependencies for mmpose
``` batch
pip install -U openmim
mim install mmengine
mim install "mmcv>-2.0.0"
mim install "mmpose>=1.0.0"
```

## Additional dependecies for building the documentation
``` batch
pip install sphinx myst-parser sphinx-copybutton sphinx-markdown-tables sphinx-rtd-theme
```

## Additional dependenceis for building tests
``` batch
pip install coverage pytest xdoctest
```


# Project organization
This project is organized with *Experiments* and *a single source of networks*.

The networks and libraries are located at `src`, Experiments are located at root, starting with two digit number.

## Dependencies


## List of experiments
The experiments were run in the following sequences
1. Random Search
2. Manual Search
3. Pretrainning
4. Weight Transfers 
5. Finetuning
6. Posture Classification training 

## Additional experiments (evaluations)
7. Additional Network - transformers
8. flop and params calculations
9. Ablation studies (SPC network structures)
10. Ablation studies (SPC hyperparameters)
11. Ablation studies (Another SPC hyperparameters)
12. SVMs
13. (Not available) Coordinate Visualizations
14. post-hoc data augmentation robustness test (Ph-DART)
15. ROC ploting

